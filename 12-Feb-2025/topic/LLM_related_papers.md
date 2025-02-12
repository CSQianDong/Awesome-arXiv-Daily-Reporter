# SymGPT: Auditing Smart Contracts via Combining Symbolic Execution with Large Language Models 

**Title (ZH)**: SymGPT：结合符号执行与大型语言模型审计智能合约 

**Authors**: Shihao Xia, Mengting He, Shuai Shao, Tingting Yu, Yiying Zhang, Linhai Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.07644)  

**Abstract**: To govern smart contracts running on Ethereum, multiple Ethereum Request for Comment (ERC) standards have been developed, each having a set of rules to guide the behaviors of smart contracts. Violating the ERC rules could cause serious security issues and financial loss, signifying the importance of verifying smart contracts follow ERCs. Today's practices of such verification are to manually audit each single contract, use expert-developed program-analysis tools, or use large language models (LLMs), all of which are far from effective in identifying ERC rule violations. This paper introduces SymGPT, a tool that combines the natural language understanding of large language models (LLMs) with the formal guarantees of symbolic execution to automatically verify smart contracts' compliance with ERC rules. To develop SymGPT, we conduct an empirical study of 132 ERC rules from three widely used ERC standards, examining their content, security implications, and natural language descriptions. Based on this study, we design SymGPT by first instructing an LLM to translate ERC rules into a defined EBNF grammar. We then synthesize constraints from the formalized rules to represent scenarios where violations may occur and use symbolic execution to detect them. Our evaluation shows that SymGPT identifies 5,783 ERC rule violations in 4,000 real-world contracts, including 1,375 violations with clear attack paths for stealing financial assets, demonstrating its effectiveness. Furthermore, SymGPT outperforms six automated techniques and a security-expert auditing service, underscoring its superiority over current smart contract analysis methods. 

**Abstract (ZH)**: 为了治理运行在以太坊上的智能合约，已经开发了多种以太坊请求评论（ERC）标准，每一种标准都有一套规则来指导智能合约的行为。违反ERC规则可能会导致严重的安全问题和经济损失，这突显了验证智能合约是否遵循ERC规则的重要性。当前这种验证的做法是手动审核每个合约，使用专家开发的程序分析工具，或者使用大型语言模型（LLMs），但这些方法远不能有效地识别ERC规则的违反情况。本文介绍了SymGPT，这是一种结合了大型语言模型（LLMs）的自然语言理解和形式验证的符号执行工具，能够自动验证智能合约是否遵循ERC规则。为了开发SymGPT，我们对三个广泛使用的ERC标准中的132条规则进行了实证研究，研究了这些规则的内容、安全影响及其自然语言描述。基于这些研究，我们首先指示LLM将ERC规则翻译成定义好的EBNF语法。然后，我们从形式化的规则中综合约束条件，表示可能发生的违规场景，并使用符号执行来检测这些场景。我们的评估结果显示，SymGPT在4000个实际合约中发现了5783个ERC规则的违反情况，其中包括1375个有明确攻击路径的违规情况，展示了其效果。此外，SymGPT在六种自动化技术和安全专家审核服务方面表现更优，进一步证明了它在当前智能合约分析方法中的优越性。 

---
# Approximating Human Strategic Reasoning with LLM-Enhanced Recursive Reasoners Leveraging Multi-agent Hypergames 

**Title (ZH)**: 使用增强递归推理器和多agent超博弈相结合来近似人类战略推理 

**Authors**: Vince Trencsenyi, Agnieszka Mensfelt, Kostas Stathis  

**Link**: [PDF](https://arxiv.org/pdf/2502.07443)  

**Abstract**: LLM-driven multi-agent-based simulations have been gaining traction with applications in game-theoretic and social simulations. While most implementations seek to exploit or evaluate LLM-agentic reasoning, they often do so with a weak notion of agency and simplified architectures. We implement a role-based multi-agent strategic interaction framework tailored to sophisticated recursive reasoners, providing the means for systematic in-depth development and evaluation of strategic reasoning. Our game environment is governed by the umpire responsible for facilitating games, from matchmaking through move validation to environment management. Players incorporate state-of-the-art LLMs in their decision mechanism, relying on a formal hypergame-based model of hierarchical beliefs. We use one-shot, 2-player beauty contests to evaluate the recursive reasoning capabilities of the latest LLMs, providing a comparison to an established baseline model from economics and data from human experiments. Furthermore, we introduce the foundations of an alternative semantic measure of reasoning to the k-level theory. Our experiments show that artificial reasoners can outperform the baseline model in terms of both approximating human behaviour and reaching the optimal solution. 

**Abstract (ZH)**: 基于LLM的多代理系统模拟在博弈理论和社会模拟领域的应用正逐渐受到关注。尽管大多数实现试图利用或评估LLM代理推理能力，但它们通常基于一种弱代理概念和简化架构。我们实现了一个基于角色的多代理战略互动框架，旨在适应复杂的递归推理者，提供系统深入开发和评估战略推理的手段。我们的游戏环境由裁判员管理，负责从匹配玩家到验证移动和环境管理的整个游戏流程。玩家在其决策机制中采用最先进的LLM，并依赖于基于形式化超博弈层次信仰模型。我们使用一次性两人的美丽竞赛来评估最新LLM的递归推理能力，提供了与经济学中的传统基准模型及人类实验数据的对比。此外，我们还引入了一种替代性语义推理度量的基础，该度量扩展了k级理论。实验结果表明，人工推理者在接近人类行为和达到最优解方面均能超越基准模型。 

---
# LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters! 

**Title (ZH)**: 大型语言模型可以从演示中轻松学会推理！结构，而不是内容，才是关键！ 

**Authors**: Dacheng Li, Shiyi Cao, Tyler Griggs, Shu Liu, Xiangxi Mo, Shishir G. Patil, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2502.07374)  

**Abstract**: Large reasoning models (LRMs) tackle complex reasoning problems by following long chain-of-thoughts (Long CoT) that incorporate reflection, backtracking, and self-validation. However, the training techniques and data requirements to elicit Long CoT remain poorly understood. In this work, we find that a Large Language model (LLM) can effectively learn Long CoT reasoning through data-efficient supervised fine-tuning (SFT) and parameter-efficient low-rank adaptation (LoRA). With just 17k long CoT training samples, the Qwen2.5-32B-Instruct model achieves significant improvements on a wide range of math and coding benchmarks, including 56.7% (+40.0%) on AIME 2024 and 57.0% (+8.1%) on LiveCodeBench, competitive to the proprietary o1-preview model's score of 44.6% and 59.1%. More importantly, we find that the structure of Long CoT is critical to the learning process, whereas the content of individual reasoning steps has minimal impact. Perturbations affecting content, such as training on incorrect samples or removing reasoning keywords, have little impact on performance. In contrast, structural modifications that disrupt logical consistency in the Long CoT, such as shuffling or deleting reasoning steps, significantly degrade accuracy. For example, a model trained on Long CoT samples with incorrect answers still achieves only 3.2% lower accuracy compared to training with fully correct samples. These insights deepen our understanding of how to elicit reasoning capabilities in LLMs and highlight key considerations for efficiently training the next generation of reasoning models. This is the academic paper of our previous released Sky-T1-32B-Preview model. Codes are available at this https URL. 

**Abstract (ZH)**: 大型推理模型（LRMs）通过遵循长推理链（Long CoT）来解决复杂的问题，这种长推理链结合了反思、回溯和自我验证。然而，如何通过训练技术及数据需求来激发长推理链的过程仍然不太清楚。在本研究中，我们发现，一个大型语言模型（LLM）可以通过高效的数据监督微调（SFT）和参数高效的小秩适应（LoRA）有效地学习长推理链推理。仅使用17000个长推理链训练样本，Qwen2.5-32B-Instruct模型在一系列数学和编程基准测试中取得了显著的改进，包括在AIME 2024上的得分提升至56.7% (+40.0%)，在LiveCodeBench上的得分提升至57.0% (+8.1%)，这与专有模型o1-preview的得分为44.6%和59.1%相当。更重要的是，我们发现长推理链的结构对学习过程至关重要，而单个推理步骤的内容则几乎没有影响。影响内容的扰动，如使用错误样本进行训练或删除推理关键词，对性能的影响甚微。相反，那些破坏长推理链逻辑一致性的结构修改，如打乱或删除推理步骤，显著降低了准确性。例如，即使在错误答案的长推理链样本上训练的模型，其准确性也仅比完全正确的样本降低了3.2%。这些发现加深了我们对激发LLMs推理能力的理解，并突显了高效训练下一代推理模型时的关键考虑因素。这是我们之前发布的Sky-T1-32B-Preview模型的学术论文。代码可从以下链接获取：this https URL。 

---
# When More is Less: Understanding Chain-of-Thought Length in LLMs 

**Title (ZH)**: 当更多变为更少：理解大语言模型中推理链长度的影响 

**Authors**: Yuyang Wu, Yifei Wang, Tianqi Du, Stefanie Jegelka, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.07266)  

**Abstract**: Chain-of-thought (CoT) reasoning enhances the multi-step reasoning capabilities of large language models (LLMs) by breaking complex tasks into smaller, manageable sub-tasks. Researchers have been exploring ways to guide models to generate more complex CoT processes to improve the reasoning ability of LLMs, such as long CoT and the test-time scaling law. However, for most models and tasks, does an increase in CoT length consistently lead to improved reasoning accuracy? In this paper, we observe a nuanced relationship: as the number of reasoning steps increases, performance initially improves but eventually decreases. To understand this phenomenon, we provide a piece of evidence that longer reasoning processes are increasingly susceptible to noise. We theoretically prove the existence of an optimal CoT length and derive a scaling law for this optimal length based on model capability and task difficulty. Inspired by our theory, we conduct experiments on both synthetic and real world datasets and propose Length-filtered Vote to alleviate the effects of excessively long or short CoTs. Our findings highlight the critical need to calibrate CoT length to align with model capabilities and task demands, offering a principled framework for optimizing multi-step reasoning in LLMs. 

**Abstract (ZH)**: 链式推理（CoT）通过将复杂任务分解为更小、更易管理的子任务，增强大型语言模型（LLMs）的多步推理能力。研究人员一直在探索引导模型生成更复杂的CoT过程的方法，以提高LLMs的推理能力，例如长链式推理和测试时的扩展定律。然而，对于大多数模型和任务而言，CoT长度的增加是否始终会导致推理准确性的提高？在本文中，我们观察到复杂的关系：随着推理步骤的增加，性能起初会提高，但最终会下降。为了理解这一现象，我们提供了一个证据，即较长的推理过程越来越容易受到噪声的影响。我们从理论上证明了存在最优的CoT长度，并根据模型能力和任务难度导出了最优长度的扩展定律。受到我们理论的启发，我们在合成和真实世界数据集上进行了实验，并提出使用“长度筛选投票”来缓解过长或过短的CoT的影响。我们的研究结果强调了根据模型能力和任务需求调整CoT长度的必要性，为优化LLMs的多步推理提供了一种原则性的框架。 

---
# Bag of Tricks for Inference-time Computation of LLM Reasoning 

**Title (ZH)**: 以下是从推理时计算大语言模型推理的技巧集 

**Authors**: Fan Liu, Wenshuo Chao, Naiqiang Tan, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.07191)  

**Abstract**: With the advancement of large language models (LLMs), solving complex reasoning tasks has gained increasing attention. Inference-time computation methods (e.g., Best-of-N, beam search, et al.) are particularly valuable as they can enhance reasoning performance without modifying model parameters or requiring additional training. However, these techniques come with implementation challenges, and most existing methods remain at the proof-of-concept stage with limited practical adoption due to their computational complexity and varying effectiveness across different tasks. In this paper, we investigate and benchmark diverse inference-time computation strategies across reasoning tasks of varying complexity. Since most current methods rely on a proposer-verifier pipeline that first generates candidate solutions (e.g., reasoning solutions) and then selects the best one based on reward signals (e.g., RLHF rewards, process rewards), our research focuses on optimizing both candidate solution generation (e.g., instructing prompts, hyperparameters such as temperature and top-p) and reward mechanisms (e.g., self-evaluation, reward types). Through extensive experiments (more than 20,000 A100-80G GPU hours with over 1,000 experiments) across a variety of models (e.g., Llama, Qwen, and Mistral families) of various sizes, our ablation studies reveal that previously overlooked strategies can significantly enhance performance (e.g., tuning temperature can improve reasoning task performance by up to 5%). Furthermore, we establish a standardized benchmark for inference-time computation by systematically evaluating six representative methods across eight reasoning tasks. These findings provide a stronger foundation for future research. The code is available at this https URL 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的不断发展，解决复杂推理任务的关注度越来越高。推理时的计算方法（如Best-of-N、束搜索等）特别有价值，因为它们可以在不修改模型参数或需要额外训练的情况下提升推理性能。然而，这些技术伴随着实施挑战，大多数现有方法仍处于概念验证阶段，因为它们的计算复杂性和在不同任务上的有效性存在差异。在本文中，我们研究并对比了多种不同复杂度的推理任务中推理时的计算策略。由于大多数当前方法依赖于提出者-验证者管道，该管道首先生成候选解决方案（例如推理解决方案），然后基于奖励信号（例如RLHF奖励、过程奖励）选择最佳方案，我们的研究集中在优化候选解决方案的生成（例如指令提示、温度和top-p等超参数）和奖励机制（例如自我评估、奖励类型）。通过在各种大小的不同模型（例如Llama、Qwen和Mistral家族）上进行超过20,000个A100-80G GPU小时的实验（超过1,000个实验），我们的消融研究揭示了一些之前被忽略的策略可以显著提升性能（例如，调整温度可以将推理任务性能提高5%）。此外，我们通过系统地评估六种代表性方法在八个推理任务上的表现，建立了推理时计算的标准基准。这些发现为未来的研究奠定了更坚实的基础。源代码可在以下链接获取：[提供建议链接] 

---
# Understanding LLMs' Fluid Intelligence Deficiency: An Analysis of the ARC Task 

**Title (ZH)**: 理解大语言模型在流体智力方面的不足：对ARC任务的分析 

**Authors**: Junjie Wu, Mo Yu, Lemao Liu, Dit-Yan Yeung, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.07190)  

**Abstract**: While LLMs have exhibited strong performance on various NLP tasks, it is noteworthy that most of these tasks rely on utilizing the vast amount of knowledge encoded in LLMs' parameters, rather than solving new problems without prior knowledge. In cognitive research, the latter ability is referred to as fluid intelligence, which is considered to be critical for assessing human intelligence. Recent research on fluid intelligence assessments has highlighted significant deficiencies in LLMs' abilities. In this paper, we analyze the challenges LLMs face in demonstrating fluid intelligence through controlled experiments, using the most representative ARC task as an example. Our study revealed three major limitations in existing LLMs: limited ability for skill composition, unfamiliarity with abstract input formats, and the intrinsic deficiency of left-to-right decoding. Our data and code can be found in this https URL. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在各种自然语言处理（NLP）任务上表现出强大的性能，值得注意的是，这些任务大多依赖于利用LLMs参数中编码的庞大知识量，而不是解决没有先验知识的新问题。在认知研究中，后一种能力被称为流体智力，被认为是对人类智力进行评估的关键。近期关于流体智力评估的研究揭示了LLMs在该方面存在的显著不足。在本文中，我们通过控制实验分析了LLMs在展示流体智力方面的挑战，以最具代表性的ARC任务为例。我们的研究揭示了现有LLMs存在的三大限制：技能组合能力有限、不熟悉抽象输入格式以及从左到右解码的固有缺陷。相关数据和代码可以在以下链接中找到：[此处链接]。 

---
# WHODUNIT: Evaluation benchmark for culprit detection in mystery stories 

**Title (ZH)**: WHODUNIT：推理侦探故事中凶手检测评估基准 

**Authors**: Kshitij Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2502.07747)  

**Abstract**: We present a novel data set, WhoDunIt, to assess the deductive reasoning capabilities of large language models (LLM) within narrative contexts. Constructed from open domain mystery novels and short stories, the dataset challenges LLMs to identify the perpetrator after reading and comprehending the story. To evaluate model robustness, we apply a range of character-level name augmentations, including original names, name swaps, and substitutions with well-known real and/or fictional entities from popular discourse. We further use various prompting styles to investigate the influence of prompting on deductive reasoning accuracy.
We conduct evaluation study with state-of-the-art models, specifically GPT-4o, GPT-4-turbo, and GPT-4o-mini, evaluated through multiple trials with majority response selection to ensure reliability. The results demonstrate that while LLMs perform reliably on unaltered texts, accuracy diminishes with certain name substitutions, particularly those with wide recognition. This dataset is publicly available here. 

**Abstract (ZH)**: 我们提出了一种新的数据集，WhoDunIt，用于评估大型语言模型（LLM）在叙事情境中的演绎推理能力。该数据集源自开放领域的侦探小说和短篇故事，旨在让LLM在阅读和理解故事后识别出罪犯。为了评估模型的鲁棒性，我们应用了一系列基于字符级别的名字增强方法，包括原始名字、名字替换以及使用广为人知的真实或虚构实体的代换。我们还使用了多种提示方式，以研究提示对演绎推理准确度的影响。

我们使用最新的模型，特别是GPT-4o、GPT-4-turbo和GPT-4o-mini，进行了评估研究。评估通过多次试验并选择多数响应来确保可靠性。结果表明，虽然LLM在未修改的文本上表现可靠，但在某些名字替换的情况下，准确度会有所下降，尤其是在那些广泛认可的名字替换中。本数据集目前已公开发布，可供查阅。 

---
# Goedel-Prover: A Frontier Model for Open-Source Automated Theorem Proving 

**Title (ZH)**: Gödel-Prover：开源自动定理证明的前沿模型 

**Authors**: Yong Lin, Shange Tang, Bohan Lyu, Jiayun Wu, Hongzhou Lin, Kaiyu Yang, Jia Li, Mengzhou Xia, Danqi Chen, Sanjeev Arora, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.07640)  

**Abstract**: We introduce Goedel-Prover, an open-source large language model (LLM) that achieves the state-of-the-art (SOTA) performance in automated formal proof generation for mathematical problems. The key challenge in this field is the scarcity of formalized math statements and proofs, which we tackle in the following ways. We train statement formalizers to translate the natural language math problems from Numina into formal language (Lean 4), creating a dataset of 1.64 million formal statements. LLMs are used to check that the formal statements accurately preserve the content of the original natural language problems. We then iteratively build a large dataset of formal proofs by training a series of provers. Each prover succeeds in proving many statements that the previous ones could not, and these new proofs are added to the training set for the next prover. The final prover outperforms all existing open-source models in whole-proof generation. On the miniF2F benchmark, it achieves a 57.6% success rate (Pass@32), exceeding the previous best open-source model by 7.6%. On PutnamBench, Goedel-Prover successfully solves 7 problems (Pass@512), ranking first on the leaderboard. Furthermore, it generates 29.7K formal proofs for Lean Workbook problems, nearly doubling the 15.7K produced by earlier works. 

**Abstract (ZH)**: 我们介绍了Goedel-Prover，这是一个开源的大规模语言模型（LLM），在数学问题自动形式证明方面达到了目前的最先进（SOTA）性能。该领域的一个关键挑战是形式化数学陈述和证明数据的稀缺性，我们通过以下方式来应对这一挑战。我们训练了陈述形式化器，将Numina中的自然语言数学问题翻译成形式语言（Lean 4），创建了一个包含164万个形式化陈述的数据集。使用大语言模型来检查这些形式化陈述是否准确地保留了原始自然语言问题的内容。然后，我们通过训练一系列证明器迭代构建一个大规模的形式证明数据集。每个证明器都能够证明前一个证明器无法证明的许多陈述，这些新的证明被添加到下一个证明器的训练集中。最终的证明器在整体定理生成方面超过了所有现有的开源模型。在miniF2F基准测试中，它实现了57.6%的成功率（Pass@32），超过了之前最好的开源模型7.6%。在PutnamBench上，Goedel-Prover成功解决了7个问题（Pass@512），排名第一。此外，它为Lean Workbook问题生成了29,700个形式证明，几乎是之前工作生成的15,700个的两倍。 

---
# Forget What You Know about LLMs Evaluations - LLMs are Like a Chameleon 

**Title (ZH)**: 忽略你对大语言模型评估的认识——大语言模型如同一种变色龙 

**Authors**: Nurit Cohen-Inger, Yehonatan Elisha, Bracha Shapira, Lior Rokach, Seffi Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2502.07445)  

**Abstract**: Large language models (LLMs) often appear to excel on public benchmarks, but these high scores may mask an overreliance on dataset-specific surface cues rather than true language understanding. We introduce the Chameleon Benchmark Overfit Detector (C-BOD), a meta-evaluation framework that systematically distorts benchmark prompts via a parametric transformation and detects overfitting of LLMs. By rephrasing inputs while preserving their semantic content and labels, C-BOD exposes whether a model's performance is driven by memorized patterns. Evaluated on the MMLU benchmark using 26 leading LLMs, our method reveals an average performance degradation of 2.15% under modest perturbations, with 20 out of 26 models exhibiting statistically significant differences. Notably, models with higher baseline accuracy exhibit larger performance differences under perturbation, and larger LLMs tend to be more sensitive to rephrasings indicating that both cases may overrely on fixed prompt patterns. In contrast, the Llama family and models with lower baseline accuracy show insignificant degradation, suggesting reduced dependency on superficial cues. Moreover, C-BOD's dataset- and model-agnostic design allows easy integration into training pipelines to promote more robust language understanding. Our findings challenge the community to look beyond leaderboard scores and prioritize resilience and generalization in LLM evaluation. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在公开基准测试中往往表现出色，但这些高分可能掩盖了模型对特定数据集表面特征的过度依赖，而非真正理解语言。我们引入了变色龙基准过拟合检测器（C-BOD），这是一种元评估框架，通过参数化转换系统地扭曲基准提示，并检测LLMs的过拟合。通过在保持语义内容和标签不变的情况下重新措辞输入，C-BOD揭示了模型性能是否由记忆化的模式驱动。在MMLU基准测试上评估26个领先的LLM模型时，我们的方法在适度扰动下揭示了平均性能下降2.15%，其中26个模型中有20个模型在统计上表现出显著差异。值得注意的是，基线准确率较高的模型在扰动下的性能差异较大，而较大的LLM对重新措辞更为敏感，这表明这两种情况可能过度依赖固定提示模式。相比之下，Llama家族模型和基线准确率较低的模型在扰动下的性能下降不显著，表明其对表面特征的依赖性较低。此外，C-BOD的设计既不依赖于特定的数据集，也不依赖于特定的模型，这使其可以轻松集成到训练管道中，促进更稳健的语言理解。我们的研究结果挑战了研究社区仅依赖排行榜得分的做法，并强调了在LLM评估中优先考虑韧性和泛化的重要性。 

---
# On Iterative Evaluation and Enhancement of Code Quality Using GPT-4o 

**Title (ZH)**: 使用GPT-4迭代评估和提升代码质量的研究 

**Authors**: Rundong Liu, Andre Frade, Amal Vaidya, Maxime Labonne, Marcus Kaiser, Bismayan Chakrabarti, Jonathan Budd, Sean Moran  

**Link**: [PDF](https://arxiv.org/pdf/2502.07399)  

**Abstract**: This paper introduces CodeQUEST, a novel framework leveraging Large Language Models (LLMs) to iteratively evaluate and enhance code quality across multiple dimensions, including readability, maintainability, efficiency, and security. The framework is divided into two main components: an Evaluator that assesses code quality across ten dimensions, providing both quantitative scores and qualitative summaries, and an Optimizer that iteratively improves the code based on the Evaluator's feedback. Our study demonstrates that CodeQUEST can effectively and robustly evaluate code quality, with its assessments aligning closely with established code quality metrics. Through a series of experiments using a curated dataset of Python and JavaScript examples, CodeQUEST demonstrated significant improvements in code quality, achieving a mean relative percentage improvement of 52.6%. The framework's evaluations were validated against a set of proxy metrics comprising of Pylint Score, Radon Maintainability Index, and Bandit output logs, showing a meaningful correlation. This highlights the potential of LLMs in automating code quality evaluation and improvement processes, presenting a significant advancement toward enhancing software development practices. The code implementation of the framework is available at: this https URL. 

**Abstract (ZH)**: 本文介绍了CodeQUEST，这是一种利用大型语言模型（LLMs）迭代评估和提升代码质量的新框架，从可读性、可维护性、效率和安全性等多个维度对代码质量进行了评估和增强。该框架分为两个主要组成部分：评估器（Evaluator）和优化器（Optimizer）。评估器通过十个维度来评估代码质量，提供定量评分和定性总结，而优化器则根据评估器的反馈逐次改进代码。我们的研究证明，CodeQUEST能够有效地且稳健地评估代码质量，其评估结果与现有的代码质量指标高度一致。通过使用精心策划的Python和JavaScript示例数据集进行的一系列实验，CodeQUEST在代码质量方面取得了显著改善，平均相对百分比提高率为52.6%。框架的评估结果还与Pylint评分、Radon可维护性指数和Bandit输出日志等代理指标进行了验证，显示了显著的相关性。这突显了LLMs在自动化代码质量评估和改进过程方面的潜力，代表了改进软件开发实践的一个重要进展。该框架的代码实现可在以下链接获取：[this https URL]。 

---
# Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering 

**Title (ZH)**: 通过有效数据过滤使大型语言模型更遵循指令并减少虚构内容 

**Authors**: Shuzheng Si, Haozhe Zhao, Gang Chen, Cheng Gao, Yuzhuo Bai, Zhitong Wang, Kaikai An, Kangyang Luo, Chen Qian, Fanchao Qi, Baobao Chang, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.07340)  

**Abstract**: Training LLMs on data that contains unfamiliar knowledge during the instruction tuning stage can make LLMs overconfident and encourage hallucinations. To address this challenge, we introduce a novel framework, NOVA, which identifies high-quality data that aligns well with the LLM's learned knowledge to reduce hallucinations. NOVA includes Internal Consistency Probing (ICP) and Semantic Equivalence Identification (SEI) to measure how familiar the LLM is with instruction data. Specifically, ICP evaluates the LLM's understanding of the given instruction by calculating the tailored consistency among multiple self-generated responses. SEI further assesses the familiarity of the LLM with the target response by comparing it to the generated responses, using the proposed semantic clustering and well-designed voting strategy. Finally, we introduce an expert-aligned reward model, considering characteristics beyond just familiarity to enhance data quality. By considering data quality and avoiding unfamiliar data, we can utilize the selected data to effectively align LLMs to follow instructions and hallucinate less. Extensive experiments and analysis show that NOVA significantly reduces hallucinations and allows LLMs to maintain a strong ability to follow instructions. 

**Abstract (ZH)**: 在指令调优阶段使用包含未 Familiar Knowledge 的数据训练语言模型会使语言模型过于自信并促进幻觉的产生。为解决这一挑战，我们引入了一种新颖的框架 NOVA，该框架通过识别与语言模型所学知识高度一致的高质量数据来减少幻觉。NOVA 包含内部一致性探针（ICP）和语义等价性识别（SEI），用于衡量语言模型对指令数据的熟悉程度。具体而言，ICP 通过计算多个自动生成响应之间的定制一致性来评估语言模型对给定指令的理解。SEI 进一步通过将其与生成的响应进行比较，使用提出的语义聚类和精心设计的投票策略来评估语言模型与目标响应的熟悉程度。最后，我们引入了一个专家对齐的奖励模型，考虑熟悉度之外的其他特征以提升数据质量。通过关注数据质量和避免使用未熟悉的数据，我们可以利用精选数据有效地使语言模型遵循指令并减少幻觉。广泛的实验和分析表明，NOVA 显著减少了幻觉，使语言模型能够保持强大的遵循指令能力。 

---
# Refine Knowledge of Large Language Models via Adaptive Contrastive Learning 

**Title (ZH)**: 通过自适应对比学习精炼大型语言模型的知识 

**Authors**: Yinghui Li, Haojing Huang, Jiayi Kuang, Yangning Li, Shu-Yu Guo, Chao Qu, Xiaoyu Tan, Hai-Tao Zheng, Ying Shen, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.07184)  

**Abstract**: How to alleviate the hallucinations of Large Language Models (LLMs) has always been the fundamental goal pursued by the LLMs research community. Looking through numerous hallucination-related studies, a mainstream category of methods is to reduce hallucinations by optimizing the knowledge representation of LLMs to change their output. Considering that the core focus of these works is the knowledge acquired by models, and knowledge has long been a central theme in human societal progress, we believe that the process of models refining knowledge can greatly benefit from the way humans learn. In our work, by imitating the human learning process, we design an Adaptive Contrastive Learning strategy. Our method flexibly constructs different positive and negative samples for contrastive learning based on LLMs' actual mastery of knowledge. This strategy helps LLMs consolidate the correct knowledge they already possess, deepen their understanding of the correct knowledge they have encountered but not fully grasped, forget the incorrect knowledge they previously learned, and honestly acknowledge the knowledge they lack. Extensive experiments and detailed analyses on widely used datasets demonstrate the effectiveness of our method. 

**Abstract (ZH)**: 如何缓解大规模语言模型（LLMs）的幻觉一直是LLMs研究领域追求的根本目标。通过对众多相关研究的综述，主流的方法之一是通过优化LLMs的知识表示来减少其输出中的幻觉现象。考虑到这些工作的核心在于模型所获取的知识，而知识一直是人类社会进步中的关键主题，我们相信模型提炼知识的过程可以从人类的学习方式中受益匪浅。在我们的工作中，通过模仿人类的学习过程，我们设计了一种自适应对比学习策略。该方法根据LLMs实际掌握的知识，灵活构建不同的正样本和负样本进行对比学习，从而帮助LLMs巩固其已掌握的正确知识，加深对已接触但尚未完全理解的正确知识的理解，忘记已经学过的错误知识，并诚实地承认其知识的不足。广泛使用的数据集上的实验结果和详细分析表明了该方法的有效性。 

---
# Rethinking Fine-Tuning when Scaling Test-Time Compute: Limiting Confidence Improves Mathematical Reasoning 

**Title (ZH)**: 当扩展测试时计算量时重新思考微调：限制置信度提高数学推理能力 

**Authors**: Feng Chen, Allan Raventos, Nan Cheng, Surya Ganguli, Shaul Druckmann  

**Link**: [PDF](https://arxiv.org/pdf/2502.07154)  

**Abstract**: Recent progress in large language models (LLMs) highlights the power of scaling test-time compute to achieve strong performance on complex tasks, such as mathematical reasoning and code generation. This raises a critical question: how should model training be modified to optimize performance under a subsequent test-time compute strategy and budget? To explore this, we focus on pass@N, a simple test-time strategy that searches for a correct answer in $N$ independent samples. We show, surprisingly, that training with cross-entropy (CE) loss can be ${\it misaligned}$ with pass@N in that pass@N accuracy ${\it decreases}$ with longer training. We explain the origins of this misalignment in terms of model overconfidence induced by CE, and experimentally verify our prediction of overconfidence as an impediment to scaling test-time compute via pass@N. Furthermore we suggest a principled, modified training loss that is better aligned to pass@N by limiting model confidence and rescuing pass@N test performance. Our algorithm demonstrates improved mathematical reasoning on MATH and MiniF2F benchmarks under several scenarios: (1) providing answers to math questions; and (2) proving theorems by searching over proof trees of varying shapes. Overall our work underscores the importance of co-designing two traditionally separate phases of LLM development: training-time protocols and test-time search and reasoning strategies. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的发展凸显了扩大测试计算规模以在复杂任务（如数学推理和代码生成）上实现强大性能的能力。这引发了关键问题：如何在后续测试计算策略和预算的优化下修改模型训练方法？为探索这一问题，我们专注于pass@N这种简单的测试计算策略，该策略在N个独立样本中搜索正确答案。我们发现，令人惊讶的是，使用交叉熵（CE）损失的训练可能导致pass@N准确性随训练时间延长而下降。我们从CE导致的模型过自信出发，解释了这种不一致性，并通过实验验证了过自信作为pass@N测试计算扩展的阻碍。此外，我们提出了一种原则性的修改训练损失函数，该函数通过限制模型自信并恢复pass@N测试性能与pass@N更好地对齐。我们的算法在MATH和MiniF2F基准测试中显示出了在多种场景下的改进：(1) 回答数学问题；(2) 通过搜索不同形状的证明树来证明定理。总体而言，我们的工作强调了需要重新设计传统分离的LLM开发阶段：训练时间和测试时间搜索与推理策略的重要性。 

---
# Cardiverse: Harnessing LLMs for Novel Card Game Prototyping 

**Title (ZH)**: Cardiverse: 利用大型语言模型进行新颖卡片游戏原型设计 

**Authors**: Danrui Li, Sen Zhang, Sam S. Sohn, Kaidong Hu, Muhammad Usman, Mubbasir Kapadia  

**Link**: [PDF](https://arxiv.org/pdf/2502.07128)  

**Abstract**: The prototyping of computer games, particularly card games, requires extensive human effort in creative ideation and gameplay evaluation. Recent advances in Large Language Models (LLMs) offer opportunities to automate and streamline these processes. However, it remains challenging for LLMs to design novel game mechanics beyond existing databases, generate consistent gameplay environments, and develop scalable gameplay AI for large-scale evaluations. This paper addresses these challenges by introducing a comprehensive automated card game prototyping framework. The approach highlights a graph-based indexing method for generating novel game designs, an LLM-driven system for consistent game code generation validated by gameplay records, and a gameplay AI constructing method that uses an ensemble of LLM-generated action-value functions optimized through self-play. These contributions aim to accelerate card game prototyping, reduce human labor, and lower barriers to entry for game developers. 

**Abstract (ZH)**: 计算机游戏，特别是纸牌游戏的原型设计需要大量的创造性构思和游戏玩法评估的人力投入。最近大型语言模型（LLMs）的进步为自动化和简化这些过程提供了机会。然而，LLMs仍然难以设计超越现有数据库的新游戏规则，生成一致的游戏环境，并为大规模评估开发可扩展的游戏AI。本文通过引入一个全面的自动化纸牌游戏原型设计框架来应对这些挑战。该方法强调基于图的索引方法以生成新颖的游戏设计，使用游戏记录验证的大型语言模型驱动的系统一致生成游戏代码，并通过自我博弈优化的大型语言模型生成的动作-价值函数构建游戏AI。这些贡献旨在加速纸牌游戏原型设计、减少人力投入，并降低游戏开发者的门槛。 

---
# Kernels of Selfhood: GPT-4o shows humanlike patterns of cognitive consistency moderated by free choice 

**Title (ZH)**: 自我内核：GPT-4o 显示出由自由选择调节的人类一致认知模式 

**Authors**: Steven A. Lehr, Ketan S. Saichandran, Eddie Harmon-Jones, Nykko Vitali, Mahzarin R. Banaji  

**Link**: [PDF](https://arxiv.org/pdf/2502.07088)  

**Abstract**: Large Language Models (LLMs) show emergent patterns that mimic human cognition. We explore whether they also mirror other, less deliberative human psychological processes. Drawing upon classical theories of cognitive consistency, two preregistered studies tested whether GPT-4o changed its attitudes toward Vladimir Putin in the direction of a positive or negative essay it wrote about the Russian leader. Indeed, GPT displayed patterns of attitude change mimicking cognitive consistency effects in humans. Even more remarkably, the degree of change increased sharply when the LLM was offered an illusion of choice about which essay (positive or negative) to write. This result suggests that GPT-4o manifests a functional analog of humanlike selfhood, although how faithfully the chatbot's behavior reflects the mechanisms of human attitude change remains to be understood. 

**Abstract (ZH)**: 大型语言模型（LLMs）展现出类似人类认知的新兴模式。我们探讨它们是否也反映其他更为非深思熟虑的人类心理过程。根据认知一致性经典的理论，我们进行了两项预先注册的研究，测试了GPT-4在撰写关于俄罗斯领导人普京的正面或负面文章之后，其对普京的态度是否发生了相应的改变。结果确实显示，GPT 的态度改变模式模仿了人类认知一致性效应。更令人惊讶的是，当LLM被提供了一个关于是要撰写正面文章还是负面文章的假象选择时，其态度改变的程度显著增加。这一结果表明，GPT-4o表现出人类相似自我功能的模拟，但聊天机器人的行为如何忠实地反映人类态度变化的机制仍有待进一步理解。 

---
# IRepair: An Intent-Aware Approach to Repair Data-Driven Errors in Large Language Models 

**Title (ZH)**: IRepair：一种基于意图的数据驱动错误修复方法在大规模语言模型中的应用 

**Authors**: Sayem Mohammad Imtiaz, Astha Singh, Fraol Batole, Hridesh Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2502.07072)  

**Abstract**: Not a day goes by without hearing about the impressive feats of large language models (LLMs), and equally, not a day passes without hearing about their challenges. LLMs are notoriously vulnerable to biases in their dataset, leading to issues such as toxicity. While domain-adaptive training has been employed to mitigate these issues, these techniques often address all model parameters indiscriminately during the repair process, resulting in poor repair quality and reduced model versatility. In this paper, we introduce a novel dynamic slicing-based intent-aware LLM repair strategy, IRepair. This approach selectively targets the most error-prone sections of the model for repair. Specifically, we propose dynamically slicing the model's most sensitive layers that require immediate attention, concentrating repair efforts on those areas. This method enables more effective repairs with potentially less impact on the model's overall performance by altering a smaller portion of the model. We evaluated our technique on three models from the GPT2 and GPT-Neo families, with parameters ranging from 800M to 1.6B, in a toxicity mitigation setup. Our results show that IRepair repairs errors 43.6% more effectively while causing 46% less disruption to general performance compared to the closest baseline, direct preference optimization. Our empirical analysis also reveals that errors are more concentrated in a smaller section of the model, with the top 20% of layers exhibiting 773% more error density than the remaining 80\%. This highlights the need for selective repair. Additionally, we demonstrate that a dynamic selection approach is essential for addressing errors dispersed throughout the model, ensuring a robust and efficient repair. 

**Abstract (ZH)**: 几乎每天都能听到关于大型语言模型（LLMs）令人印象深刻的表现，同样地，几乎每天也能听到它们所面临挑战的报道。LLMs 通常因其数据集中的偏见而易受攻击，导致诸如毒性等问题。虽然领域适配训练已被用来缓解这些问题，但这些技术在修复过程中往往不分青红皂白地处理所有模型参数，导致修复质量较差且模型的灵活性降低。在本文中，我们提出了一种新颖的动态切片为基础、意识目标的LLM修复策略IRepair。该方法有针对性地修复模型中最易出错的部分。具体而言，我们提出动态切片模型中最敏感、需要立即关注的层，将修复精力集中在这些区域上。这种方法通过改变较小的部分模型来实现更有效的修复，同时对模型整体性能的影响较小。我们在毒性缓解设置中评估了我们的技术，测试了来自GPT2和GPT-Neo家族的三个参数范围分别为800M至1.6B的模型。结果显示，IRepair 在修复错误方面比最近的基线（直接偏好优化）有效 43.6%，并且导致的总体性能影响减少 46%。我们的实证分析还表明，错误主要集中在模型的一个较小部分，其中顶尖的20%层的错误密度比剩余的80%高773%。这突显了选择性修复的必要性。此外，我们证明了动态选择方法对于处理模型中分散的错误至关重要，从而确保修复的稳健性和效率。 

---
# Large Language Models in Software Security: A Survey of Vulnerability Detection Techniques and Insights 

**Title (ZH)**: 大型语言模型在软件安全中的应用：漏洞检测技术综述与见解 

**Authors**: Ze Sheng, Zhicheng Chen, Shuning Gu, Heqing Huang, Guofei Gu, Jeff Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.07049)  

**Abstract**: Large Language Models (LLMs) are emerging as transformative tools for software vulnerability detection, addressing critical challenges in the security domain. Traditional methods, such as static and dynamic analysis, often falter due to inefficiencies, high false positive rates, and the growing complexity of modern software systems. By leveraging their ability to analyze code structures, identify patterns, and generate repair sugges- tions, LLMs, exemplified by models like GPT, BERT, and CodeBERT, present a novel and scalable approach to mitigating vulnerabilities. This paper provides a detailed survey of LLMs in vulnerability detection. It examines key aspects, including model architectures, application methods, target languages, fine-tuning strategies, datasets, and evaluation metrics. We also analyze the scope of current research problems, highlighting the strengths and weaknesses of existing approaches. Further, we address challenges such as cross-language vulnerability detection, multimodal data integration, and repository-level analysis. Based on these findings, we propose solutions for issues like dataset scalability, model interpretability, and applications in low-resource scenarios. Our contributions are threefold: (1) a systematic review of how LLMs are applied in vulnerability detection; (2) an analysis of shared patterns and differences across studies, with a unified framework for understanding the field; and (3) a summary of key challenges and future research directions. This work provides valuable insights for advancing LLM-based vulnerability detection. We also maintain and regularly update latest selected paper on this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）正在成为软件漏洞检测的变革性工具，有望解决安全领域中的关键挑战。传统的静态和动态分析方法由于效率低下、误报率高和现代软件系统日益增加的复杂性而常常失效。通过利用其分析代码结构、识别模式和生成修复建议的能力，LLMs，如GPT、BERT和CodeBERT等模型，提供了一种新颖且可扩展的方法来减轻漏洞。本文详细综述了LLMs在漏洞检测领域的应用。它探讨了关键方面，包括模型架构、应用方法、目标语言、微调策略、数据集和评估指标。我们还分析了当前研究问题的范围，指出了现有方法的优势和不足之处。此外，我们还讨论了跨语言漏洞检测、多模态数据整合和仓库级分析等挑战。基于这些发现，我们提出了关于数据集可扩展性、模型可解释性和低资源场景应用问题的解决方案。我们的贡献体现在三个方面：(1) 系统综述LLMs在漏洞检测中的应用；(2) 对研究方法中的共同模式和差异进行分析，并构建一个统一的理解该领域的框架；(3) 总结关键挑战和未来的研究方向。这项工作为推进基于LLM的漏洞检测提供了宝贵的见解。我们也维护并定期更新了最新的相关研究论文，地址为：https://www.alipay.com 

---
# Scalable and Ethical Insider Threat Detection through Data Synthesis and Analysis by LLMs 

**Title (ZH)**: 通过大规模语言模型进行数据合成与分析以实现可扩展且伦理的内部威胁检测 

**Authors**: Haywood Gelman, John D. Hastings  

**Link**: [PDF](https://arxiv.org/pdf/2502.07045)  

**Abstract**: Insider threats wield an outsized influence on organizations, disproportionate to their small numbers. This is due to the internal access insiders have to systems, information, and infrastructure. %One example of this influence is where anonymous respondents submit web-based job search site reviews, an insider threat risk to organizations. Signals for such risks may be found in anonymous submissions to public web-based job search site reviews. This research studies the potential for large language models (LLMs) to analyze and detect insider threat sentiment within job site reviews. Addressing ethical data collection concerns, this research utilizes synthetic data generation using LLMs alongside existing job review datasets. A comparative analysis of sentiment scores generated by LLMs is benchmarked against expert human scoring. Findings reveal that LLMs demonstrate alignment with human evaluations in most cases, thus effectively identifying nuanced indicators of threat sentiment. The performance is lower on human-generated data than synthetic data, suggesting areas for improvement in evaluating real-world data. Text diversity analysis found differences between human-generated and LLM-generated datasets, with synthetic data exhibiting somewhat lower diversity. Overall, the results demonstrate the applicability of LLMs to insider threat detection, and a scalable solution for insider sentiment testing by overcoming ethical and logistical barriers tied to data acquisition. 

**Abstract (ZH)**: 内部威胁在组织中的影响力远超过其人数，与内部人员对系统、信息和基础设施的访问权限有关。这种威胁的一个例子是匿名回答者提交基于网络的求职网站评论，这构成了对组织的一种内部威胁风险。这类风险的信号可能出现在公共求职网站评论中的匿名提交中。本研究探讨了大规模语言模型（LLMs）分析和检测求职网站评论中内部威胁情感的可能性。为了应对伦理数据采集问题，本研究使用LLMs生成的合成数据以及现有的求职评论数据集。通过将LLMs生成的情感评分与专家人类评分进行对比分析，基准测试显示LLMs大多数情况下能够与人类评估保持一致，有效地识别威胁情感的细微指标。在人类生成的数据上，LLMs的表现低于合成数据，这表明在评估真实世界数据方面存在改进空间。文本多样性的分析发现，人类生成的数据集和LLMs生成的数据集之间存在差异，合成数据集的多样性较低。总体而言，结果表明LLMs在内部威胁检测中的适用性，并提供了一种克服数据获取伦理和物流障碍的可扩展解决方案，以测试内部人员的情感。 

---
# Automated Consistency Analysis of LLMs 

**Title (ZH)**: 自动一致性分析模型（或：自动大型语言模型的一致性分析） 

**Authors**: Aditya Patwardhan, Vivek Vaidya, Ashish Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2502.07036)  

**Abstract**: Generative AI (Gen AI) with large language models (LLMs) are being widely adopted across the industry, academia and government. Cybersecurity is one of the key sectors where LLMs can be and/or are already being used. There are a number of problems that inhibit the adoption of trustworthy Gen AI and LLMs in cybersecurity and such other critical areas. One of the key challenge to the trustworthiness and reliability of LLMs is: how consistent an LLM is in its responses?
In this paper, we have analyzed and developed a formal definition of consistency of responses of LLMs. We have formally defined what is consistency of responses and then develop a framework for consistency evaluation. The paper proposes two approaches to validate consistency: self-validation, and validation across multiple LLMs. We have carried out extensive experiments for several LLMs such as GPT4oMini, GPT3.5, Gemini, Cohere, and Llama3, on a security benchmark consisting of several cybersecurity questions: informational and situational. Our experiments corroborate the fact that even though these LLMs are being considered and/or already being used for several cybersecurity tasks today, they are often inconsistent in their responses, and thus are untrustworthy and unreliable for cybersecurity. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的生成型人工智能（Gen AI）在工业、学术界和政府领域的广泛应用，网络安全领域是其中一个重要应用领域。然而，信任和可靠性的问题是阻碍Gen AI和LLMs在网络安全及其他关键领域广泛应用的重要障碍之一。影响LLMs信任度和可靠性的关键挑战之一是：LLMs在回应问题时的一致性如何？

在本文中，我们对LLMs响应的一致性进行了分析并提出了正式定义。首先，我们定义了响应一致性的含义，然后建立了一个一致性评估框架。本文提出了两种验证一致性的方法：自我验证和跨多个LLMs的验证。我们在GPT4oMini、GPT3.5、Gemini、Cohere和Llama3等多个LLMs上进行了大量实验，使用了一个网络安全基准，其中包括多个网络安全问题，如信息性和情境性问题。我们的实验结果证实，尽管这些LLMs被用于多种网络安全任务，但它们在回应问题时经常不一致，因此在网络安全方面缺乏可信性和可靠性。 

---
# Leveraging GPT-4o Efficiency for Detecting Rework Anomaly in Business Processes 

**Title (ZH)**: 利用GPT-4的效率检测业务流程中的返工异常 

**Authors**: Mohammad Derakhshan, Paolo Ceravolo, Fatemeh Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.06918)  

**Abstract**: This paper investigates the effectiveness of GPT-4o-2024-08-06, one of the Large Language Models (LLM) from OpenAI, in detecting business process anomalies, with a focus on rework anomalies. In our study, we developed a GPT-4o-based tool capable of transforming event logs into a structured format and identifying reworked activities within business event logs. The analysis was performed on a synthetic dataset designed to contain rework anomalies but free of loops. To evaluate the anomaly detection capabilities of GPT 4o-2024-08-06, we used three prompting techniques: zero-shot, one-shot, and few-shot. These techniques were tested on different anomaly distributions, namely normal, uniform, and exponential, to identify the most effective approach for each case. The results demonstrate the strong performance of GPT-4o-2024-08-06. On our dataset, the model achieved 96.14% accuracy with one-shot prompting for the normal distribution, 97.94% accuracy with few-shot prompting for the uniform distribution, and 74.21% accuracy with few-shot prompting for the exponential distribution. These results highlight the model's potential as a reliable tool for detecting rework anomalies in event logs and how anomaly distribution and prompting strategy influence the model's performance. 

**Abstract (ZH)**: 本文探讨了来自OpenAI的大语言模型（LLM）GPT-4o-2024-08-06在检测业务流程异常中的有效性，特别是重做异常。在本研究中，我们开发了一个基于GPT-4o的工具，能够将事件日志转换为结构化格式，并识别业务事件日志中的重做活动。分析是在一个合成数据集上进行的，该数据集包含重做异常但没有循环。为了评估GPT 4o-2024-08-06在检测异常方面的能力，我们使用了三种提示技术：零样本、单样本和少样本。这些技术分别应用于不同的异常分布，即标准分布、均匀分布和指数分布，以确定每种情况下最有效的策略。结果表明，GPT-4o-2024-08-06表现出色。在我们的数据集上，模型通过单样本提示在标准分布下达到96.14%的准确率，通过少样本提示在均匀分布下达到97.94%的准确率，而在指数分布下通过少样本提示达到74.21%的准确率。这些结果突显了该模型作为检测事件日志中重做异常的可靠工具的潜力，并且表明异常分布和提示策略对模型性能的影响。 

---
# Enabling Autoregressive Models to Fill In Masked Tokens 

**Title (ZH)**: 使得自回归模型能够填充掩码令牌 

**Authors**: Daniel Israel, Aditya Grover, Guy Van den Broeck  

**Link**: [PDF](https://arxiv.org/pdf/2502.06901)  

**Abstract**: Historically, LLMs have been trained using either autoregressive (AR) or masked language modeling (MLM) objectives, with AR models gaining dominance in recent years. However, AR models are inherently incapable of masked infilling, which is the ability to predict masked tokens between past and future context. In contrast, MLM models suffer from intrinsic computational inefficiencies during both training and inference that hinder their scalability. This work introduces MARIA (Masked and Autoregressive Infilling Architecture), a novel approach that leverages the strengths of both paradigms to achieve state-of-the-art masked infilling performance. MARIA combines a pre-trained MLM and AR model by training a linear decoder that takes their concatenated hidden states as input. This minimal modification enables the AR model to perform infilling while retaining its inherent advantages in terms of faster inference with KV caching. Our results demonstrate that MARIA significantly outperforms existing methods, namely discrete diffusion models, on masked infilling tasks. 

**Abstract (ZH)**: 历史上，大型语言模型（LLMs）通常通过自回归（AR）或掩码语言建模（MLM）的目标进行训练，近年来自回归模型逐渐占据主导地位。然而，自回归模型本质上无法实现掩码填充能力，即预测过去和未来语境之间的掩码词的能力。相比之下，掩码语言建模模型在训练和推理过程中固有的计算效率低下问题阻碍了其可扩展性。本文介绍了一种新的方法——MARIA（Masked and Autoregressive Infilling Architecture），它结合了两种范式的优点，实现了最先进的掩码填充性能。MARIA通过训练一个线性解码器来结合预训练的MLM模型和AR模型，该解码器将两者拼接后的隐藏状态作为输入。这一最小的修改使得AR模型能够进行填充操作，同时保留其在带有KV缓存的快速推理方面的固有优势。我们的实验结果表明，MARIA在掩码填充任务上显著优于现有方法，特别是离散扩散模型。 

---
# Large Language Models for In-File Vulnerability Localization Can Be "Lost in the End" 

**Title (ZH)**: 大型语言模型在文件内漏洞定位中可能会“迷失终点” 

**Authors**: Francesco Sovrano, Adam Bauer, Alberto Bacchelli  

**Link**: [PDF](https://arxiv.org/pdf/2502.06898)  

**Abstract**: Recent advancements in artificial intelligence have enabled processing of larger inputs, leading everyday software developers to increasingly rely on chat-based large language models (LLMs) like GPT-3.5 and GPT-4 to detect vulnerabilities across entire files, not just within functions. This new development practice requires researchers to urgently investigate whether commonly used LLMs can effectively analyze large file-sized inputs, in order to provide timely insights for software developers and engineers about the pros and cons of this emerging technological trend. Hence, the goal of this paper is to evaluate the effectiveness of several state-of-the-art chat-based LLMs, including the GPT models, in detecting in-file vulnerabilities. We conducted a costly investigation into how the performance of LLMs varies based on vulnerability type, input size, and vulnerability location within the file. To give enough statistical power to our study, we could only focus on the three most common (as well as dangerous) vulnerabilities: XSS, SQL injection, and path traversal. Our findings indicate that the effectiveness of LLMs in detecting these vulnerabilities is strongly influenced by both the location of the vulnerability and the overall size of the input. Specifically, regardless of the vulnerability type, LLMs tend to significantly (p < .05) underperform when detecting vulnerabilities located toward the end of larger files, a pattern we call the 'lost-in-the-end' effect. Finally, to further support software developers and practitioners, we also explored the optimal input size for these LLMs and presented a simple strategy for identifying it, which can be applied to other models and vulnerability types. Eventually, we show how adjusting the input size can lead to significant improvements in LLM-based vulnerability detection, with an average recall increase of over 37% across all models. 

**Abstract (ZH)**: 近年来，人工智能的最新进展使处理大规模输入成为可能，这促使日常软件开发者越来越多地依赖基于聊天的大语言模型（LLMs），如GPT-3.5和GPT-4，来检测整个文件中的漏洞，而不仅仅是文件中的函数。这种新的开发实践要求研究人员亟需调查常用的LLMs是否能够有效地分析大型文件输入，以便为软件开发者和工程师提供关于这一新兴技术趋势的优势和局限性及时见解。因此，本文旨在评估几种最先进的基于聊天的LLMs，包括GPT模型，检测文件内漏洞的有效性。我们针对不同类型的漏洞、输入规模和文件内的漏洞位置进行了耗时的研究，以确保研究有足够的统计能力。由于我们关注的是最常见的也是最危险的三种漏洞：XSS、SQL注入和路径遍历。研究结果表明，LLMs检测这些漏洞的有效性受漏洞位置和输入整体规模的影响。具体来说，无论漏洞类型如何，当检测位于较大文件末尾的漏洞时，LLMs往往会显著（p < .05）表现出不佳的表现，我们称之为“末尾丢失”效应。最后，为了进一步支持软件开发者和从业人员，我们还探讨了这些LLMs的最佳输入规模，并提出了一种简单策略来识别，该策略可以应用于其他模型和漏洞类型。最终，我们展示了调整输入规模可以显著提高基于LLMs的漏洞检测效果，各种模型的召回率平均提高超过37%。 

---
# Learning Conformal Abstention Policies for Adaptive Risk Management in Large Language and Vision-Language Models 

**Title (ZH)**: 面向大型语言和多模态模型的自适应风险管理的收敛可信区间弃权策略学习 

**Authors**: Sina Tayebati, Divake Kumar, Nastaran Darabi, Dinithi Jayasuriya, Ranganath Krishnan, Amit Ranjan Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2502.06884)  

**Abstract**: Large Language and Vision-Language Models (LLMs/VLMs) are increasingly used in safety-critical applications, yet their opaque decision-making complicates risk assessment and reliability. Uncertainty quantification (UQ) helps assess prediction confidence and enables abstention when uncertainty is high. Conformal prediction (CP), a leading UQ method, provides statistical guarantees but relies on static thresholds, which fail to adapt to task complexity and evolving data distributions, leading to suboptimal trade-offs in accuracy, coverage, and informativeness. To address this, we propose learnable conformal abstention, integrating reinforcement learning (RL) with CP to optimize abstention thresholds dynamically. By treating CP thresholds as adaptive actions, our approach balances multiple objectives, minimizing prediction set size while maintaining reliable coverage. Extensive evaluations across diverse LLM/VLM benchmarks show our method outperforms Least Ambiguous Classifiers (LAC) and Adaptive Prediction Sets (APS), improving accuracy by up to 3.2%, boosting AUROC for hallucination detection by 22.19%, enhancing uncertainty-guided selective generation (AUARC) by 21.17%, and reducing calibration error by 70%-85%. These improvements hold across multiple models and datasets while consistently meeting the 90% coverage target, establishing our approach as a more effective and flexible solution for reliable decision-making in safety-critical applications. The code is available at: {this https URL}. 

**Abstract (ZH)**: 大型语言模型（LLMs）和多模态语言-视觉模型（VLMs）在安全关键应用中的使用越来越广泛，但它们透明度低的决策过程使得风险评估和可靠性分析变得复杂。不确定性量化（UQ）有助于评估预测的信心，并在不确定性高时允许避免做出决策。可靠的不确定性估计方法之一是校准预测（Conformal Prediction, CP），它提供了统计保证，但依赖于静态阈值，这些阈值无法适应任务复杂度和数据分布的变化，导致在准确性和覆盖率之间存在次优权衡。为了解决这个问题，我们提出了一种可学习的校准预测避免策略，将强化学习（Reinforcement Learning, RL）与CP相结合，以动态优化避免阈值。通过将CP阈值视为可学习的动作，我们的方法能够平衡多重目标，在保持可靠的覆盖率的同时最小化预测集的大小。在多种LLM/VLM基准测试中的详尽评估表明，我们的方法优于最少含糊类别分类器（Least Ambiguous Classifiers, LAC）和自适应预测集（Adaptive Prediction Sets, APS），准确率最高可提高3.2%，对于幻觉检测的AUROC提高了22.19%，对于不确定性引导的选择性生成（AUARC）提高了21.17%，并且减少了70%-85%的校准误差。这些改进在多个模型和数据集上保持一致，并且始终达到90%的覆盖率目标，证明了我们的方法在安全关键应用中更有效、更灵活的决策制定方案。代码可从以下链接获得：{this https URL}。 

---
# Multi-Agent Simulator Drives Language Models for Legal Intensive Interaction 

**Title (ZH)**: 多agents模拟器驱动的语言模型在法律密集型交互中的应用 

**Authors**: Shengbin Yue, Ting Huang, Zheng Jia, Siyuan Wang, Shujun Liu, Yun Song, Xuanjing Huang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.06882)  

**Abstract**: Large Language Models (LLMs) have significantly advanced legal intelligence, but the scarcity of scenario data impedes the progress toward interactive legal scenarios. This paper introduces a Multi-agent Legal Simulation Driver (MASER) to scalably generate synthetic data by simulating interactive legal scenarios. Leveraging real-legal case sources, MASER ensures the consistency of legal attributes between participants and introduces a supervisory mechanism to align participants' characters and behaviors as well as addressing distractions. A Multi-stage Interactive Legal Evaluation (MILE) benchmark is further constructed to evaluate LLMs' performance in dynamic legal scenarios. Extensive experiments confirm the effectiveness of our framework. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在法律智能方面取得了显著进展，但场景数据的稀缺性阻碍了交互式法律场景的发展。本文介绍了一种多智能体法律模拟驱动器（MASER），通过模拟交互式法律场景来大规模生成合成数据。利用真实的法律案例源，MASER 确保了参与者之间法律属性的一致性，并引入了一个监督机制来对齐参与者的性格和行为，同时解决干扰问题。进一步构建了一个多阶段交互式法律评估（MILE）基准，以评估在动态法律场景中LLMs 的表现。广泛的实验验证了我们框架的有效性。 

---
# Mix Data or Merge Models? Balancing the Helpfulness, Honesty, and Harmlessness of Large Language Model via Model Merging 

**Title (ZH)**: 混合数据还是合并模型？通过模型合并平衡大规模语言模型的帮助性、诚实性和无害性 

**Authors**: Jinluan Yang, Dingnan Jin, Anke Tang, Li Shen, Didi Zhu, Zhengyu Chen, Daixin Wang, Qing Cui, Zhiqiang Zhang, Jun Zhou, Fei Wu, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2502.06876)  

**Abstract**: Achieving balanced alignment of large language models (LLMs) in terms of Helpfulness, Honesty, and Harmlessness (3H optimization) constitutes a cornerstone of responsible AI, with existing methods like data mixture strategies facing limitations including reliance on expert knowledge and conflicting optimization signals. While model merging offers a promising alternative by integrating specialized models, its potential for 3H optimization remains underexplored. This paper establishes the first comprehensive benchmark for model merging in 3H-aligned LLMs, systematically evaluating 15 methods (12 training-free merging and 3 data mixture techniques) across 10 datasets associated with 5 annotation dimensions, 2 LLM families, and 2 training paradigms. Our analysis reveals three pivotal insights: (i) previously overlooked collaborative/conflicting relationships among 3H dimensions, (ii) the consistent superiority of model merging over data mixture approaches in balancing alignment trade-offs, and (iii) the critical role of parameter-level conflict resolution through redundant component pruning and outlier mitigation. Building on these findings, we propose R-TSVM, a Reweighting-enhanced Task Singular Vector Merging method that incorporates outlier-aware parameter weighting and sparsity-adaptive rank selection strategies adapted to the heavy-tailed parameter distribution and sparsity for LLMs, further improving LLM alignment across multiple evaluations. Our models will be available at this https URL. 

**Abstract (ZH)**: 实现大型语言模型（LLMs）在帮助性、诚实性与无害性（3H优化）方面的平衡对负责任的人工智能至关重要。现有的方法，如数据混合策略，面临着依赖专家知识和优化信号冲突的局限。同时，模型合并作为一个有前景的替代方案，通过整合专业模型而具有潜在优势，但其在3H优化方面的潜力仍需进一步探索。本文建立了首个针对3H对齐的大语言模型中的模型合并基准，系统地评估了15种方法（12种无需训练的合并方法和3种数据混合技术）在10个关联5个注释维度、2个LLM家族和2种训练范式的数据集上的性能。我们的分析揭示了三个关键的见解：(i) 前期被忽视的3H维度之间的合作/冲突关系；(ii) 模型合并方法在平衡对齐权衡方面始终优于数据混合方法；(iii) 参数级别的冲突解决机制，包括冗余组件的修剪和异常值的缓解，发挥着关键作用。基于这些发现，我们提出了一种增强任务特征向量合并方法——R-TSVM（重权重任务特征向量合并），该方法结合了对异常值敏感的参数权重和根据LLM的重尾参数分布及其稀疏性自适应选择的秩选择策略，进一步提高了LLM的对齐程度。我们的模型将在以下网址提供：[请填写具体网址]。 

---
# Beyond Vision: How Large Language Models Interpret Facial Expressions from Valence-Arousal Values 

**Title (ZH)**: 超越视觉：大型语言模型如何根据正负值和唤醒值解释面部表情 

**Authors**: Vaibhav Mehra, Guy Laban, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2502.06875)  

**Abstract**: Large Language Models primarily operate through text-based inputs and outputs, yet human emotion is communicated through both verbal and non-verbal cues, including facial expressions. While Vision-Language Models analyze facial expressions from images, they are resource-intensive and may depend more on linguistic priors than visual understanding. To address this, this study investigates whether LLMs can infer affective meaning from dimensions of facial expressions-Valence and Arousal values, structured numerical representations, rather than using raw visual input. VA values were extracted using Facechannel from images of facial expressions and provided to LLMs in two tasks: (1) categorizing facial expressions into basic (on the IIMI dataset) and complex emotions (on the Emotic dataset) and (2) generating semantic descriptions of facial expressions (on the Emotic dataset). Results from the categorization task indicate that LLMs struggle to classify VA values into discrete emotion categories, particularly for emotions beyond basic polarities (e.g., happiness, sadness). However, in the semantic description task, LLMs produced textual descriptions that align closely with human-generated interpretations, demonstrating a stronger capacity for free text affective inference of facial expressions. 

**Abstract (ZH)**: 大型语言模型主要通过文本形式的输入和输出进行操作，而人类情感的表达则通过言语和非言语线索，包括面部表情等进行传递。尽管视觉语言模型可以从图像中分析面部表情，但这些模型在资源消耗方面较为密集，并且可能更依赖于语言先验而非视觉理解。为了解决这一问题，本研究探讨了大型语言模型是否能够从面部表情的情感维度——正负价值（Valence）和唤醒度（Arousal）的结构化数值表示中推断情感意义，而不需要使用原始的视觉输入。面部表情的VA值通过Facechannel从图像中提取，并提供给大型语言模型完成两个任务：（1）在IIMI数据集上将面部表情分类为基本情绪和复杂情绪，在Emotic数据集上进行这项任务；（2）在Emotic数据集上生成面部表情的语义描述。分类任务的结果表明，大型语言模型在将VA值分类为离散的情绪类别时面临困难，尤其是在基本极性情绪之外（例如，快乐、悲伤）的情况。然而，在语义描述任务中，大型语言模型生成的文本描述与人类生成的解释高度一致，显示出更强的自由文本情感推断能力。 

---
# Group Reasoning Emission Estimation Networks 

**Title (ZH)**: 群体推理排放估算网络 

**Authors**: Yanming Guo, Xiao Qian, Kevin Credit, Jin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.06874)  

**Abstract**: Accurate greenhouse gas (GHG) emission reporting is critical for governments, businesses, and investors. However, adoption remains limited particularly among small and medium enterprises due to high implementation costs, fragmented emission factor databases, and a lack of robust sector classification methods. To address these challenges, we introduce Group Reasoning Emission Estimation Networks (GREEN), an AI-driven carbon accounting framework that standardizes enterprise-level emission estimation, constructs a large-scale benchmark dataset, and leverages a novel reasoning approach with large language models (LLMs). Specifically, we compile textual descriptions for 20,850 companies with validated North American Industry Classification System (NAICS) labels and align these with an economic model of carbon intensity factors. By reframing sector classification as an information retrieval task, we fine-tune Sentence-BERT models using a contrastive learning loss. To overcome the limitations of single-stage models in handling thousands of hierarchical categories, we propose a Group Reasoning method that ensembles LLM classifiers based on the natural NAICS ontology, decomposing the task into multiple sub-classification steps. We theoretically prove that this approach reduces classification uncertainty and computational complexity. Experiments on 1,114 NAICS categories yield state-of-the-art performance (83.68% Top-1, 91.47% Top-10 accuracy), and case studies on 20 companies report a mean absolute percentage error (MAPE) of 45.88%. The project is available at: this https URL. 

**Abstract (ZH)**: 准确的温室气体（GHG）排放申报对于政府、企业和投资者至关重要。然而，由于实施成本高、排放因子数据库碎片化以及缺乏稳健的行业分类方法，中小企业在采用这些措施方面仍存在限制。为克服这些挑战，我们引入了Group Reasoning Emission Estimation Networks（GREEN），一个基于人工智能的碳核算框架，该框架统一了企业级排放估计，构建了大规模基准数据集，并利用大型语言模型（LLMs）的新型推理方法。具体来说，我们为20,850家具有验证过的北美行业分类系统（NAICS）标签的公司编制了文本描述，并将其与碳强度因素的经济模型对齐。通过将行业分类重新构想为信息检索任务，我们使用对比学习损失微调了Sentence-BERT模型。为了解决单一阶段模型处理数千个层级类别时的限制，我们提出了Group Reasoning方法，基于自然的NAICS本体论集合LLM分类器，将任务分解为多个子分类步骤。我们理论证明了这种方法降低了分类不确定性并减少了计算复杂度。在1,114个NAICS类别上的实验结果达到了最先进的性能（Top-1精度83.68%，Top-10精度91.47%），针对20家公司的案例研究表明，平均绝对百分比误差（MAPE）为45.88%。该项目可在以下链接查看：this https URL。 

---
# Multimodal Cognitive Reframing Therapy via Multi-hop Psychotherapeutic Reasoning 

**Title (ZH)**: 多模态认知重构疗法通过多跳心理治疗推理 

**Authors**: Subin Kim, Hoonrae Kim, Heejin Do, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.06873)  

**Abstract**: Previous research has revealed the potential of large language models (LLMs) to support cognitive reframing therapy; however, their focus was primarily on text-based methods, often overlooking the importance of non-verbal evidence crucial in real-life therapy. To alleviate this gap, we extend the textual cognitive reframing to multimodality, incorporating visual clues. Specifically, we present a new dataset called Multi Modal-Cognitive Support Conversation (M2CoSC), which pairs each GPT-4-generated dialogue with an image that reflects the virtual client's facial expressions. To better mirror real psychotherapy, where facial expressions lead to interpreting implicit emotional evidence, we propose a multi-hop psychotherapeutic reasoning approach that explicitly identifies and incorporates subtle evidence. Our comprehensive experiments with both LLMs and vision-language models (VLMs) demonstrate that the VLMs' performance as psychotherapists is significantly improved with the M2CoSC dataset. Furthermore, the multi-hop psychotherapeutic reasoning method enables VLMs to provide more thoughtful and empathetic suggestions, outperforming standard prompting methods. 

**Abstract (ZH)**: 以往的研究揭示了大型语言模型（LLMs）在支持认知重构疗法方面的潜力；然而，这些研究主要集中在基于文本的方法上，往往忽视了在实际疗法中至关重要的非语言证据的重要性。为了解决这一问题，我们将文本认知重构扩展到多模态领域，引入了视觉线索。具体而言，我们提出了一种名为多模态认知支持对话（M2CoSC）的新数据集，该数据集每包含一个GPT-4生成的对话，还配有一张反映虚拟来访者面部表情的照片。为了更好地模拟实际的心理咨询过程，其中面部表情能够引导对隐含的情绪证据的解读，我们提出了一种多跳心理疗法推理方法，以明确识别并融入细微的证据。我们的全面实验表明，使用M2CoSC数据集，视觉语言模型（VLMs）在心理治疗中的表现得到了显著提升。此外，多跳心理疗法推理方法使VLMs能够提供更为周到和富有同情心的建议，超越了标准的提示方法。 

---
# Knowledge Graph-Guided Retrieval Augmented Generation 

**Title (ZH)**: 知识图谱引导的检索增强生成 

**Authors**: Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.06864)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a promising technology for addressing hallucination issues in the responses generated by large language models (LLMs). Existing studies on RAG primarily focus on applying semantic-based approaches to retrieve isolated relevant chunks, which ignore their intrinsic relationships. In this paper, we propose a novel Knowledge Graph-Guided Retrieval Augmented Generation (KG$^2$RAG) framework that utilizes knowledge graphs (KGs) to provide fact-level relationships between chunks, improving the diversity and coherence of the retrieved results. Specifically, after performing a semantic-based retrieval to provide seed chunks, KG$^2$RAG employs a KG-guided chunk expansion process and a KG-based chunk organization process to deliver relevant and important knowledge in well-organized paragraphs. Extensive experiments conducted on the HotpotQA dataset and its variants demonstrate the advantages of KG$^2$RAG compared to existing RAG-based approaches, in terms of both response quality and retrieval quality. 

**Abstract (ZH)**: 检索增强生成（RAG）被认为是解决大规模语言模型（LLMs）生成响应中幻觉问题的一种有前途的技术。现有关于RAG的研究主要集中在应用基于语义的方法从孤立的相关片段中检索信息，但忽略了这些片段之间的内在关系。本文提出了一种新型的知识图谱指导的检索增强生成（KG²RAG）框架，利用知识图谱（KGs）提供片段间的事实级关系，从而提高检索结果的多样性和连贯性。具体而言，KG²RAG 在基于语义的检索提供种子片段之后，通过一个知识图谱指导的片段扩展过程和一个基于知识图谱的片段组织过程，将相关且重要的知识以有条理的段落形式呈现。在HotpotQA数据集及其变体上进行的广泛实验表明，KG²RAG 在响应质量和检索质量方面都优于现有的RAG方法。 

---
# LLM-Supported Natural Language to Bash Translation 

**Title (ZH)**: LLM支持的自然语言到Bash脚本转换 

**Authors**: Finnian Westenfelder, Erik Hemberg, Miguel Tulla, Stephen Moskal, Una-May O'Reilly, Silviu Chiricescu  

**Link**: [PDF](https://arxiv.org/pdf/2502.06858)  

**Abstract**: The Bourne-Again Shell (Bash) command-line interface for Linux systems has complex syntax and requires extensive specialized knowledge. Using the natural language to Bash command (NL2SH) translation capabilities of large language models (LLMs) for command composition circumvents these issues. However, the NL2SH performance of LLMs is difficult to assess due to inaccurate test data and unreliable heuristics for determining the functional equivalence of Bash commands. We present a manually verified test dataset of 600 instruction-command pairs and a training dataset of 40,939 pairs, increasing the size of previous datasets by 441% and 135%, respectively. Further, we present a novel functional equivalence heuristic that combines command execution with LLM evaluation of command outputs. Our heuristic can determine the functional equivalence of two Bash commands with 95% confidence, a 16% increase over previous heuristics. Evaluation of popular LLMs using our test dataset and heuristic demonstrates that parsing, in-context learning, in-weight learning, and constrained decoding can improve NL2SH accuracy by up to 32%. Our findings emphasize the importance of dataset quality, execution-based evaluation and translation method for advancing NL2SH translation. Our code is available at this https URL 

**Abstract (ZH)**: 以下是翻译成中文的内容，符合学术规范：

Linux系统中的Bourne-Again Shell（Bash）命令行界面具有复杂的语法，需要广泛的专业知识。通过大型语言模型（LLMs）的自然语言到Bash命令（NL2SH）转换能力来进行命令组合可以规避这些问题。然而，由于测试数据不准确以及确定Bash命令功能等价性的校验规则不可靠，因此LLMs的NL2SH性能评估颇具挑战。本文提出了一组经过手动验证的测试数据集，包含600条指令-命令对，以及一个训练数据集，包含40,939对指令-命令对，分别将之前的数据集规模扩大了441%和135%。此外，本文还提出了一种新颖的功能等价性校验规则，结合了命令执行与LLM对命令输出的评价。该规则能够以95%的置信度确定两个Bash命令的功能等价性，相比于之前的方法，准确度提高了16%。使用本文提供的测试数据集和校验规则对流行的大规模语言模型进行评估表明，解析、上下文学习、权重学习和约束解码可以将NL2SH的准确性提高最多32%。研究发现强调了数据集质量、基于执行的评估以及翻译方法对于推进NL2SH翻译的重要性。相关代码可在以下网址获取：[[链接]] 

---
# Can Large Language Models Understand Intermediate Representations? 

**Title (ZH)**: 大型语言模型能否理解中间表示？ 

**Authors**: Hailong Jiang, Jianfeng Zhu, Yao Wan, Bo Fang, Hongyu Zhang, Ruoming Jin, Qiang Guan  

**Link**: [PDF](https://arxiv.org/pdf/2502.06854)  

**Abstract**: Intermediate Representations (IRs) are essential in compiler design and program analysis, yet their comprehension by Large Language Models (LLMs) remains underexplored. This paper presents a pioneering empirical study to investigate the capabilities of LLMs, including GPT-4, GPT-3, Gemma 2, LLaMA 3.1, and Code Llama, in understanding IRs. We analyze their performance across four tasks: Control Flow Graph (CFG) reconstruction, decompilation, code summarization, and execution reasoning. Our results indicate that while LLMs demonstrate competence in parsing IR syntax and recognizing high-level structures, they struggle with control flow reasoning, execution semantics, and loop handling. Specifically, they often misinterpret branching instructions, omit critical IR operations, and rely on heuristic-based reasoning, leading to errors in CFG reconstruction, IR decompilation, and execution reasoning. The study underscores the necessity for IR-specific enhancements in LLMs, recommending fine-tuning on structured IR datasets and integration of explicit control flow models to augment their comprehension and handling of IR-related tasks. 

**Abstract (ZH)**: 中间表示（Intermediate Representations，IRs）在编译器设计和程序分析中起着至关重要的作用，但大型语言模型（Large Language Models，LLMs）对其的理解程度仍较少被探索。本文进行了一项开创性的实证研究，以探讨LLMs（包括GPT-4、GPT-3、Gemma 2、LLaMA 3.1和Code Llama）在理解IRs方面的能力。我们在这四项任务——控制流图（Control Flow Graph，CFG）重构、反编译、代码总结和执行推理——中分析了它们的性能。研究结果表明，虽然LLMs在解析IR语法和识别高层次结构方面表现出色，但在控制流推理、执行语义和循环处理方面存在困难。具体而言，它们经常误读分支指令，省略关键的IR操作，并依赖基于启发式的推理，导致CFG重构、IR反编译和执行推理中的错误。该研究强调了在LLMs中对IR特定增强的必要性，建议在结构化的IR数据集上进行微调，并集成显式控制流模型，以增强它们对IR相关任务的理解和处理能力。 

---
# Policy Guided Tree Search for Enhanced LLM Reasoning 

**Title (ZH)**: 政策引导的树搜索方法以增强语言模型推理能力 

**Authors**: Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.06813)  

**Abstract**: Despite their remarkable capabilities, large language models often struggle with tasks requiring complex reasoning and planning. While existing approaches like Chain-of-Thought prompting and tree search techniques show promise, they are limited by their reliance on predefined heuristics and computationally expensive exploration strategies. We propose Policy-Guided Tree Search (PGTS), a framework that combines reinforcement learning with structured tree exploration to efficiently navigate reasoning paths. Our key innovation is a learned policy that dynamically decides between expanding, branching, backtracking, or terminating exploration, eliminating the need for manual heuristics or exhaustive search. Experiments across mathematical reasoning, logical deduction, and planning benchmarks demonstrate that PGTS achieves superior reasoning performance while significantly reducing computational costs compared to existing methods. These results establish PGTS as a scalable and effective solution for tackling complex reasoning tasks with LLMs. 

**Abstract (ZH)**: 尽管大型语言模型具有卓越的能力，但在需要复杂推理和规划的任务中常常表现出色有限。现有的方法，如链式思考提示和树搜索技术虽然显示出前景，但由于依赖预定义的启发式方法和计算成本较高的探索策略，它们仍存在局限性。本文提出了一种结合强化学习与结构化树探索的Policy-Guided Tree Search（PGTS）框架，以高效地导航推理路径。我们的主要创新在于一个学习到的策略，该策略能够动态地决定在扩展、分叉、回溯或终止探索之间做出选择，从而消除了手动启发式或穷举搜索的需要。在数学推理、逻辑推导和规划基准测试中的实验表明，PGTS不仅在推理性能方面优于现有方法，而且在计算成本方面显著降低。这些结果验证了PGTS作为利用大规模语言模型处理复杂推理任务的可扩展且有效解决方案的有效性。 

---
# Competitive Programming with Large Reasoning Models 

**Title (ZH)**: 大规模推理模型的竞赛编程 

**Authors**: OpenAI, Ahmed El-Kishky, Alexander Wei, Andre Saraiva, Borys Minaev, Daniel Selsam, David Dohan, Francis Song, Hunter Lightman, Ignasi Clavera, Jakub Pachocki, Jerry Tworek, Lorenz Kuhn, Lukasz Kaiser, Mark Chen, Max Schwarzer, Mostafa Rohaninejad, Nat McAleese, o3 contributors, Oleg Mürk, Rhythm Garg, Rui Shu, Szymon Sidor, Vineet Kosaraju, Wenda Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.06807)  

**Abstract**: We show that reinforcement learning applied to large language models (LLMs) significantly boosts performance on complex coding and reasoning tasks. Additionally, we compare two general-purpose reasoning models - OpenAI o1 and an early checkpoint of o3 - with a domain-specific system, o1-ioi, which uses hand-engineered inference strategies designed for competing in the 2024 International Olympiad in Informatics (IOI). We competed live at IOI 2024 with o1-ioi and, using hand-crafted test-time strategies, placed in the 49th percentile. Under relaxed competition constraints, o1-ioi achieved a gold medal. However, when evaluating later models such as o3, we find that o3 achieves gold without hand-crafted domain-specific strategies or relaxed constraints. Our findings show that although specialized pipelines such as o1-ioi yield solid improvements, the scaled-up, general-purpose o3 model surpasses those results without relying on hand-crafted inference heuristics. Notably, o3 achieves a gold medal at the 2024 IOI and obtains a Codeforces rating on par with elite human competitors. Overall, these results indicate that scaling general-purpose reinforcement learning, rather than relying on domain-specific techniques, offers a robust path toward state-of-the-art AI in reasoning domains, such as competitive programming. 

**Abstract (ZH)**: 我们展示了将强化学习应用于大规模语言模型（LLMs）显著提升了复杂编码和推理任务的性能。此外，我们比较了两个通用推理模型——OpenAI的o1和o3的早期检查点——与一个特定领域的系统o1-ioi，该系统利用为2024年国际信息学奥林匹克（IOI）设计的手工工程化推理策略。在2024年IOI的比赛中，我们使用手工制作的测试时策略，o1-ioi位居第49百分位。在较为宽松的竞赛约束条件下，o1-ioi获得了金牌。然而，在评估后续模型如o3时，我们发现o3在没有依靠手工制作的领域特定策略或宽松约束的情况下也能获得金牌。我们的研究表明，尽管专门化的流水线如o1-ioi能够带来显著改进，但规模更大的通用模型o3在推理领域超越这些结果。值得注意的是，o3在2024年IOI中取得了金牌，并且在Codeforces上的排名与顶级的人类竞争对手相当。总体而言，这些结果表明，通过扩大通用强化学习的应用而非依赖于领域特定技术，为推理领域，如编程竞赛，提供了通往先进AI的稳健途径。 

---
# Logits are All We Need to Adapt Closed Models 

**Title (ZH)**: 我们需要的是仅限于适应闭源模型的逻辑值 

**Authors**: Gaurush Hiranandani, Haolun Wu, Subhojyoti Mukherjee, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2502.06806)  

**Abstract**: Many commercial Large Language Models (LLMs) are often closed-source, limiting developers to prompt tuning for aligning content generation with specific applications. While these models currently do not provide access to token logits, we argue that if such access were available, it would enable more powerful adaptation techniques beyond prompt engineering. In this paper, we propose a token-level probability reweighting framework that, given access to logits and a small amount of task-specific data, can effectively steer black-box LLMs toward application-specific content generation. Our approach views next-token prediction through the lens of supervised classification. We show that aligning black-box LLMs with task-specific data can be formulated as a label noise correction problem, leading to \emph{Plugin} model -- an autoregressive probability reweighting model that operates solely on logits. We provide theoretical justification for why reweighting logits alone is sufficient for task adaptation. Extensive experiments with multiple datasets, LLMs, and reweighting models demonstrate the effectiveness of our method, advocating for broader access to token logits in closed-source models. 

**Abstract (ZH)**: 许多商业大型语言模型（LLMs）通常是闭源的，限制了开发者仅通过提示调优来实现内容生成与特定应用的对齐。虽然当前这些模型并未提供对标记logits的访问，但我们认为，如果能够访问logits，将会使开发者能够采用超越提示工程的更强大的适应技术。在本文中，我们提出了一种标记级别概率重加权框架，该框架在获得logits和少量任务特定数据的情况下，能够有效引导黑盒LLMs生成特定应用的内容。我们的方法将下一个标记的预测视为监督分类问题。我们表明，将黑盒LLMs与任务特定数据进行对齐可以表述为标签噪声矫正问题，这导致了“插件”模型——一个仅基于logits进行自回归概率重加权的模型。我们提供了理论依据，证明仅仅重加权logits就足以适应任务需求。广泛的实验证明了我们方法的有效性，并倡导在闭源模型中更广泛地提供token logits的访问权限。 

---
# Solving the Content Gap in Roblox Game Recommendations: LLM-Based Profile Generation and Reranking 

**Title (ZH)**: 解决 Roblox 游戏推荐中的内容缺口：基于大语言模型的角色生成与重新 ranking 

**Authors**: Chen Wang, Xiaokai Wei, Yexi Jiang, Frank Ong, Kevin Gao, Xiao Yu, Zheng Hui, Se-eun Yoon, Philip Yu, Michelle Gong  

**Link**: [PDF](https://arxiv.org/pdf/2502.06802)  

**Abstract**: With the vast and dynamic user-generated content on Roblox, creating effective game recommendations requires a deep understanding of game content. Traditional recommendation models struggle with the inconsistent and sparse nature of game text features such as titles and descriptions. Recent advancements in large language models (LLMs) offer opportunities to enhance recommendation systems by analyzing in-game text data. This paper addresses two challenges: generating high-quality, structured text features for games without extensive human annotation, and validating these features to ensure they improve recommendation relevance. We propose an approach that extracts in-game text and uses LLMs to infer attributes such as genre and gameplay objectives from raw player interactions. Additionally, we introduce an LLM-based re-ranking mechanism to assess the effectiveness of the generated text features, enhancing personalization and user satisfaction. Beyond recommendations, our approach supports applications such as user engagement-based integrity detection, already deployed in production. This scalable framework demonstrates the potential of in-game text understanding to improve recommendation quality on Roblox and adapt recommendations to its unique, user-generated ecosystem. 

**Abstract (ZH)**: 在Roblox平台上，海量且动态用户生成的内容使得有效的游戏推荐需要深入理解游戏内容。传统的推荐模型难以应对游戏文本特征（如标题和描述）的不一致性和稀疏性。近年来，在大规模语言模型（LLMs）方面的进步为通过分析游戏内文本数据提升推荐系统提供了机会。本文旨在解决两个挑战：生成无需大量人工注释的高质量、结构化文本特征，以及验证这些特征能否提高推荐的相关性。我们提出了一种方法，提取游戏内文本，并使用LLMs从原始玩家互动中推断属性（如游戏类型和 gameplay 目标）。此外，我们引入了一种基于LLM的重排序机制，以评估生成的文本特征的有效性，从而增强个性化并提升用户体验。除了推荐之外，我们的方法还支持如基于用户参与度的完整性检测等应用，已在生产环境中部署。本文提出的一种可扩展框架展示了游戏内文本理解在提高Roblox上的推荐质量以及适应其独特的用户生成生态系统方面的潜力。 

---
# ETimeline: An Extensive Timeline Generation Dataset based on Large Language Model 

**Title (ZH)**: ETimeline: 一种基于大型语言模型的广泛时间线生成数据集 

**Authors**: Xiaochen Liu, Yanan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.07474)  

**Abstract**: Timeline generation is of great significance for a comprehensive understanding of the development of events over time. Its goal is to organize news chronologically, which helps to identify patterns and trends that may be obscured when viewing news in isolation, making it easier to track the development of stories and understand the interrelationships between key events. Timelines are now common in various commercial products, but academic research in this area is notably scarce. Additionally, the current datasets are in need of refinement for enhanced utility and expanded coverage. In this paper, we propose ETimeline, which encompasses over $13,000$ news articles, spanning $600$ bilingual timelines across $28$ news domains. Specifically, we gather a candidate pool of more than $120,000$ news articles and employ the large language model (LLM) Pipeline to improve performance, ultimately yielding the ETimeline. The data analysis underscores the appeal of ETimeline. Additionally, we also provide the news pool data for further research and analysis. This work contributes to the advancement of timeline generation research and supports a wide range of tasks, including topic generation and event relationships. We believe that this dataset will serve as a catalyst for innovative research and bridge the gap between academia and industry in understanding the practical application of technology services. The dataset is available at this https URL 

**Abstract (ZH)**: 时间轴生成对于全面理解事件随时间的发展具有重要意义。其目标是按时间顺序组织新闻，有助于识别单独查看新闻时可能被掩盖的模式和趋势，从而更方便地追踪故事的发展并理解关键事件之间的相互关系。目前，时间轴在各种商业产品中已经很常见，但在这一领域的学术研究相对稀缺。此外，当前的数据集需要进一步优化，以增强其实用性和覆盖范围。在本文中，我们提出了ETimeline，涵盖了超过13,000篇新闻文章，跨越了28个新闻领域的600多条双语时间轴。具体而言，我们收集了一个超过120,000篇新闻文章的候选池，并采用了大型语言模型（LLM）Pipeline以提高性能，最终产生了ETimeline。数据分析显示了ETimeline的吸引力。此外，我们还提供了新闻池数据，以供进一步研究和分析。这项工作促进了时间轴生成研究的发展，并支持各种任务，包括主题生成和事件关系分析。我们相信，这个数据集将成为创新研究的催化剂，并在理解和应用技术方面弥合学术界与工业界之间的差距。数据集可在此链接中获取：[此链接] 

---
# CreAgent: Towards Long-Term Evaluation of Recommender System under Platform-Creator Information Asymmetry 

**Title (ZH)**: CreAgent：在平台创造者信息不对称情境下的推荐系统长期评估方法探究 

**Authors**: Xiaopeng Ye, Chen Xu, Zhongxiang Sun, Jun Xu, Gang Wang, Zhenhua Dong, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.07307)  

**Abstract**: Ensuring the long-term sustainability of recommender systems (RS) emerges as a crucial issue. Traditional offline evaluation methods for RS typically focus on immediate user feedback, such as clicks, but they often neglect the long-term impact of content creators. On real-world content platforms, creators can strategically produce and upload new items based on user feedback and preference trends. While previous studies have attempted to model creator behavior, they often overlook the role of information asymmetry. This asymmetry arises because creators primarily have access to feedback on the items they produce, while platforms possess data on the entire spectrum of user feedback. Current RS simulators, however, fail to account for this asymmetry, leading to inaccurate long-term evaluations. To address this gap, we propose CreAgent, a Large Language Model (LLM)-empowered creator simulation agent. By incorporating game theory's belief mechanism and the fast-and-slow thinking framework, CreAgent effectively simulates creator behavior under conditions of information asymmetry. Additionally, we enhance CreAgent's simulation ability by fine-tuning it using Proximal Policy Optimization (PPO). Our credibility validation experiments show that CreAgent aligns well with the behaviors between real-world platform and creator, thus improving the reliability of long-term RS evaluations. Moreover, through the simulation of RS involving CreAgents, we can explore how fairness- and diversity-aware RS algorithms contribute to better long-term performance for various stakeholders. CreAgent and the simulation platform are publicly available at this https URL. 

**Abstract (ZH)**: 确保推荐系统（RS）的长期可持续性成为了一个关键问题。传统意义上的离线评估方法通常侧重于用户的即时反馈，例如点击行为，但往往忽视了内容创作者的长期影响。在现实生活中的内容平台中，创作者可以根据用户反馈和偏好趋势战略性地生产并上传新内容。虽然之前的研究尝试建模创作者的行为，但这些研究往往忽略了信息不对称的作用。这种不对称性源于创作者对其生产内容的反馈有更多接触，而平台则拥有整个用户反馈谱的数据。然而，目前的RS模拟器未能考虑到这一点不对称性，导致长期评价不够准确。为了填补这一缺口，我们提出了一种由大型语言模型（LLM）赋能的创作者模拟代理——CreAgent。通过结合博弈论的信任机制和快速思考与慢速思考框架，CreAgent有效地模拟了在信息不对称条件下的创作者行为。此外，我们通过使用近端策略优化（PPO）对CreAgent进行微调，从而增强其模拟能力。我们的信誉验证实验表明，CreAgent与实际平台和创作者的行为高度契合，从而提高了长期RS评价的可靠性。此外，通过对涉及CreAgent的RS进行模拟，我们可以探究公平性和多样化感知的算法如何促进各种利益相关者的长期性能提升。CreAgent及其模拟平台已公开可供访问：https://this-url-edited-by-ai.com 

---
# Making Language Models Robust Against Negation 

**Title (ZH)**: 使语言模型在面对否定时具有稳健性 

**Authors**: MohammadHossein Rezaei, Eduardo Blanco  

**Link**: [PDF](https://arxiv.org/pdf/2502.07717)  

**Abstract**: Negation has been a long-standing challenge for language models. Previous studies have shown that they struggle with negation in many natural language understanding tasks. In this work, we propose a self-supervised method to make language models more robust against negation. We introduce a novel task, Next Sentence Polarity Prediction (NSPP), and a variation of the Next Sentence Prediction (NSP) task. We show that BERT and RoBERTa further pre-trained on our tasks outperform the off-the-shelf versions on nine negation-related benchmarks. Most notably, our pre-training tasks yield between 1.8% and 9.1% improvement on CondaQA, a large question-answering corpus requiring reasoning over negation. 

**Abstract (ZH)**: 否定形式一直是语言模型的一个长期挑战。先前的研究表明，它们在许多自然语言理解任务中处理否定形式存在困难。在本研究中，我们提出了一种自监督方法，以提高语言模型对否定形式的鲁棒性。我们引入了一个新型任务——下一句极性预测（NSPP），以及下一句预测（NSP）任务的一个变体。我们表明，我们的任务进一步预训练的BERT和RoBERTa在九个与否定有关的基准测试中优于即用版本。最显著的是，我们的预训练任务在CondaQA上（一个需要在否定方面进行推理的大型问答语料库）分别提供了1.8%到9.1%的性能提升。 

---
# FoQA: A Faroese Question-Answering Dataset 

**Title (ZH)**: FoQA：一种 Faroese 问答数据集 

**Authors**: Annika Simonsen, Dan Saattrup Nielsen, Hafsteinn Einarsson  

**Link**: [PDF](https://arxiv.org/pdf/2502.07642)  

**Abstract**: We present FoQA, a Faroese extractive question-answering (QA) dataset with 2,000 samples, created using a semi-automated approach combining Large Language Models (LLMs) and human validation. The dataset was generated from Faroese Wikipedia articles using GPT-4-turbo for initial QA generation, followed by question rephrasing to increase complexity and native speaker validation to ensure quality. We provide baseline performance metrics for FoQA across multiple models, including LLMs and BERT, demonstrating its effectiveness in evaluating Faroese QA performance. The dataset is released in three versions: a validated set of 2,000 samples, a complete set of all 10,001 generated samples, and a set of 2,395 rejected samples for error analysis. 

**Abstract (ZH)**: 我们介绍了FoQA，这是一个包含2000个样本的远罗佛语提取式问答(QA)数据集，该数据集是通过结合大型语言模型（LLMs）和人工验证的半自动化方法创建的。数据集是从远罗佛语维基百科文章生成的，使用GPT-4-turbo进行初始的QA生成，随后通过重新表述问题来增加复杂性，并通过母语者的验证确保质量。我们提供了多个模型，包括LLMs和BERT，在FoQA上的基线性能指标，这表明了FoQA在评估远罗佛语问答性能方面的有效性。数据集提供了三个版本：一个包含2000个验证样本的版本、一个包含所有10001个生成样本的完整版本以及一个包含2395个被拒绝样本的错误分析版本。 

---
# O1 Embedder: Let Retrievers Think Before Action 

**Title (ZH)**: O1嵌入器：让检索器在行动前思考 

**Authors**: Ruin Yan, Zheng Liu, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2502.07555)  

**Abstract**: The growing power of large language models (LLMs) has revolutionized how people access and utilize information. Notably, the LLMs excel at performing fine-grained data representation, which facilitates precise retrieval of information. They also generate high-quality answers based on external references, enabling the production of useful knowledge. The recent introduction of reasoning models, like OpenAI O1 and DeepSeek R1, marks another leap forward, highlighting LLMs' ability to think progressively before delivering final answers. This breakthrough significantly improves the ability to address complex tasks, e.g., coding and math proofs.
Inspired by this progress, we aim to develop similar capabilities for retrieval models, which hold great promise for tackling critical challenges in the field, including multi-task retrieval, zero-shot retrieval, and tasks requiring intensive reasoning of complex relationships. With this motivation, we propose a novel approach called O1 Embedder, which generates useful thoughts for the input query before making retrieval for the target documents. To realize this objective, we conquer two technical difficulties. First, we design a data synthesis workflow, creating training signals for O1 Embedder by generating initial thoughts from an LLM-expert and subsequently refining them using a retrieval committee. Second, we optimize the training process, enabling a pre-trained model to be jointly fine-tuned to generate retrieval thoughts via behavior cloning and perform dense retrieval through contrastive learning. Our approach is evaluated by comprehensive experiments, where substantial improvements are achieved across 12 popular datasets, spanning both in-domain and out-of-domain scenarios. These results highlight O1 Embedder's remarkable accuracy and generalizability, paving the way for the development of next-generation IR foundation models. 

**Abstract (ZH)**: 大型语言模型（LLMs）的力量不断增强，这已彻底改变了人们获取和利用信息的方式。值得注意的是，LLMs在进行细粒度数据表示方面表现出色，从而促进了精准的信息检索。它们也能基于外部参考生成高质量的答案，使知识生产变得有用。近期引入的推理模型，如OpenAI O1和DeepSeek R1，标志着又一个重要进步，突显了LLMs具备在提供最终答案之前逐步推理的能力。这一突破显著提高了应对复杂任务的能力，例如编程和数学证明。

受到这一进展的启发，我们旨在为检索模型开发类似的能力，这些模型在处理领域内和跨领域的重要挑战方面具有巨大潜力，包括多任务检索、零样本检索以及需要对复杂关系进行深入推理的任务。为了实现这一目标，我们提出了一种名为O1 Embedder的新方法，在进行目标文档检索之前，O1 Embedder能生成输入查询的有用想法。为了实现这一目标，我们克服了两个技术难题。首先，我们设计了一个数据合成工作流程，通过生成LLM专家初稿思想并随后使用检索委员会对其进行细化，为O1 Embedder生成训练信号。其次，我们优化了训练过程，使得预先训练的模型能够通过行为克隆进行联合微调，生成检索思想，并通过对比学习进行密集检索。我们的方法在全面的实验中进行了评估，在12个流行的基准数据集中均取得了显著的改进，这些数据集涵盖了领域内和领域外的情景。这些结果展示了O1 Embedder出色的准确性和泛化能力，为下一代IR基础模型的开发铺平了道路。 

---
# Mask-Enhanced Autoregressive Prediction: Pay Less Attention to Learn More 

**Title (ZH)**: 掩码增强自回归预测：学会更多的同时关注更少 

**Authors**: Xialie Zhuang, Zhikai Jia, Jianjin Li, Zhenyu Zhang, Li Shen, Zheng Cao, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.07490)  

**Abstract**: Large Language Models (LLMs) are discovered to suffer from accurately retrieving key information. To address this, we propose Mask-Enhanced Autoregressive Prediction (MEAP), a simple yet effective training paradigm that seamlessly integrates Masked Language Modeling (MLM) into Next-Token Prediction (NTP) to enhance the latter's in-context retrieval capabilities. Specifically, MEAP first randomly masks a small fraction of input tokens and then directly performs the standard next-token prediction autoregressive using a decoder-only Transformer. MEAP eliminates the need for bidirectional attention or encoder-decoder architectures for MLM, incurring no additional computational overhead during pre-training or inference. Intensive experiments demonstrate that MEAP substantially outperforms NTP on key information retrieval and long-context reasoning tasks, while performing on par or better on commonsense reasoning tasks. The benefits of MEAP also extend to supervised fine-tuning, where it shows remarkable advantages in lost-in-the-middle scenarios, outperforming NTP by 11.77 percentage points. Our analysis indicates that MEAP's effectiveness arises from its ability to promote more distinguishable attention scores by concentrating on a reduced set of non-masked tokens. This mechanism improves the model's focus on task-relevant signals while mitigating the influence of peripheral context. These findings position MEAP as a promising training paradigm for large language models. 

**Abstract (ZH)**: 大型语言模型（LLMs）被发现存在准确检索关键信息的问题。为了解决这一问题，我们提出了Mask-Enhanced Autoregressive Prediction (MEAP)，这是一种简单而有效的训练范式，能够无缝将Masked Language Modeling (MLM) 集成到Next-Token Prediction (NTP) 中，从而增强其上下文检索能力。具体而言，MEAP 首先随机掩蔽一小部分输入词元，然后使用仅解码器的Transformer 进行标准的下一词预测自回归。MEAP 消除了需要双向注意力或编码器-解码器架构来进行MLM 的需求，在预训练和推理阶段不会增加额外的计算开销。密集的实验证明，MEAP 在关键信息检索和长上下文推理任务上的表现显著优于NTP，而在常识推理任务上的表现则不输或更优。MEAP 的优势还扩展到监督微调，它在“丢失中间信息”的场景中表现出显著优势，比NTP 高出11.77 个百分点。我们的分析表明，MEAP 的有效性来自于它能够通过集中关注未掩蔽的词元集合，促进更有区别的注意力评分。这一机制能够提高模型对任务相关信号的关注度，同时减少边缘上下文的影响。这些发现将MEAP 定位为一种有前途的大型语言模型训练范式。 

---
# Entity Linking using LLMs for Automated Product Carbon Footprint Estimation 

**Title (ZH)**: 使用大语言模型进行自动产品碳足迹估算的实体链接方法 

**Authors**: Steffen Castle, Julian Moreno Schneider, Leonhard Hennig, Georg Rehm  

**Link**: [PDF](https://arxiv.org/pdf/2502.07418)  

**Abstract**: Growing concerns about climate change and sustainability are driving manufacturers to take significant steps toward reducing their carbon footprints. For these manufacturers, a first step towards this goal is to identify the environmental impact of the individual components of their products. We propose a system leveraging large language models (LLMs) to automatically map components from manufacturer Bills of Materials (BOMs) to Life Cycle Assessment (LCA) database entries by using LLMs to expand on available component information. Our approach reduces the need for manual data processing, paving the way for more accessible sustainability practices. 

**Abstract (ZH)**: 关于气候变化和可持续发展的深切担忧正在推动制造商采取重大措施减少自身的碳足迹。对于这些制造商来说，实现这一目标的第一步是识别其产品各个组件的环境影响。我们提出了一种基于大型语言模型（LLM）的系统，利用LLM扩展现有组件信息，自动将制造商的物料清单（BOM）中的组件映射到生命周期评估（LCA）数据库条目。我们的方法减少了手动数据处理的需要，为更广泛的可持续实践打开了大门。 

---
# LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation 

**Title (ZH)**: 长文境：通过恢复蒸馏缓解长背景大规模语言模型中的短文本退化问题 

**Authors**: Zican Dong, Junyi Li, Jinhao Jiang, Mingyu Xu, Wayne Xin Zhao, Bingning Wang, Weipeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.07365)  

**Abstract**: Large language models (LLMs) have gained extended context windows through scaling positional encodings and lightweight continual pre-training. However, this often leads to degraded performance on short-text tasks, while the reasons for this degradation remain insufficiently explored. In this work, we identify two primary factors contributing to this issue: distribution drift in hidden states and attention scores, and catastrophic forgetting during continual pre-training. To address these challenges, we propose Long Context Pre-training with Restoration Distillation (LongReD), a novel approach designed to mitigate short-text performance degradation through minimizing the distribution discrepancy between the extended and original models. Besides training on long texts, LongReD distills the hidden state of selected layers from the original model on short texts. Additionally, LongReD also introduces a short-to-long distillation, aligning the output distribution on short texts with that on long texts by leveraging skipped positional indices. Experiments on common text benchmarks demonstrate that LongReD effectively preserves the model's short-text performance while maintaining comparable or even better capacity to handle long texts than baselines. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过扩展位置编码和轻量级连续预训练获得了更长的上下文窗口。然而，这常常导致其在短文本任务上的性能下降，而这些性能下降的具体原因仍缺乏充分的探讨。本文我们识别出两个主要因素导致了这一问题：隐藏状态和注意力分数的分布漂移，以及连续预训练过程中的灾难性遗忘。为解决这些问题，我们提出了一种名为Long Context Pre-training with Restoration Distillation（LongReD）的新颖方法，旨在通过最小化扩展模型和原始模型之间分布差异来缓解短文本上的性能下降。除了使用长文本进行训练，LongReD 还从原始模型中选择层的隐藏状态在短文本上进行蒸馏。此外，LongReD 还引入了短到长的蒸馏方法，通过利用跳过的位置索引使短文本上的输出分布与长文本上的输出分布对齐。在通用文本基准测试上的实验结果表明，LongReD 在保持模型短文本性能的同时，还能在处理长文本方面与基准模型具有相当甚至更好的能力。 

---
# Bridging the Evaluation Gap: Leveraging Large Language Models for Topic Model Evaluation 

**Title (ZH)**: 填补评估差距：利用大型语言模型进行主题模型评估 

**Authors**: Zhiyin Tan, Jennifer D'Souza  

**Link**: [PDF](https://arxiv.org/pdf/2502.07352)  

**Abstract**: This study presents a framework for automated evaluation of dynamically evolving topic taxonomies in scientific literature using Large Language Models (LLMs). In digital library systems, topic modeling plays a crucial role in efficiently organizing and retrieving scholarly content, guiding researchers through complex knowledge landscapes. As research domains proliferate and shift, traditional human centric and static evaluation methods struggle to maintain relevance. The proposed approach harnesses LLMs to measure key quality dimensions, such as coherence, repetitiveness, diversity, and topic-document alignment, without heavy reliance on expert annotators or narrow statistical metrics. Tailored prompts guide LLM assessments, ensuring consistent and interpretable evaluations across various datasets and modeling techniques. Experiments on benchmark corpora demonstrate the method's robustness, scalability, and adaptability, underscoring its value as a more holistic and dynamic alternative to conventional evaluation strategies. 

**Abstract (ZH)**: 本研究提出了一种利用大型语言模型（LLMs）自动评估动态演化的科学文献主题分类的框架。在数字图书馆系统中，主题建模在高效组织和检索学术内容、引导研究人员通过复杂的知识景观方面发挥着关键作用。随着研究领域的扩展和变化，传统的基于人工和静态的评估方法难以保持相关性。提出的这种方法利用LLMs衡量关键的质量维度，如一致性、重复性、多样性和主题-文档对齐，而不依赖于专家注释或狭窄的统计指标。定制化的提示语指导LLMs的评估，确保在不同数据集和建模技术下的一致性和可解释性评估。基准语料库上的实验表明该方法的鲁棒性、可扩展性和灵活性，突显了其作为比传统评估策略更具全面性和动态性的替代方法的价值。 

---
# BenchMAX: A Comprehensive Multilingual Evaluation Suite for Large Language Models 

**Title (ZH)**: BenchMAX：全面的多语言大型语言模型评估套件 

**Authors**: Xu Huang, Wenhao Zhu, Hanxu Hu, Conghui He, Lei Li, Shujian Huang, Fei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.07346)  

**Abstract**: Previous multilingual benchmarks focus primarily on simple understanding tasks, but for large language models(LLMs), we emphasize proficiency in instruction following, reasoning, long context understanding, code generation, and so on. However, measuring these advanced capabilities across languages is underexplored. To address the disparity, we introduce BenchMAX, a multi-way multilingual evaluation benchmark that allows for fair comparisons of these important abilities across languages. To maintain high quality, three distinct native-speaking annotators independently annotate each sample within all tasks after the data was machine-translated from English into 16 other languages. Additionally, we present a novel translation challenge stemming from dataset construction. Extensive experiments on BenchMAX reveal varying effectiveness of core capabilities across languages, highlighting performance gaps that cannot be bridged by simply scaling up model size. BenchMAX serves as a comprehensive multilingual evaluation platform, providing a promising test bed to promote the development of multilingual language models. The dataset and code are publicly accessible. 

**Abstract (ZH)**: 之前的多语言基准主要集中在简单的理解任务上，但对于大型语言模型（LLMs），我们更强调其在指令遵循、推理、长文理解、代码生成等方面的能力。然而，这些高级能力跨语言的衡量仍然存在不足。为解决这一问题，我们引入了BenchMAX，这是一种多语言评估基准，允许在多种语言中公平比较这些重要的能力。为了保持高质量，数据从英语翻译成16种其他语言后，三名独立的母语标注者分别对所有任务的每个样本进行标注。此外，我们还介绍了数据集构建过程中产生的新颖翻译挑战。BenchMAX 上的大量实验揭示了这些核心能力在不同语言中的不同有效性，突出了不能仅通过增加模型规模来弥合的性能差距。BenchMAX 作为一个全面的多语言评估平台，提供了一个有前景的测试环境，促进多语言语言模型的发展。该数据集和代码均已公开。 

---
# Don't Just Demo, Teach Me the Principles: A Principle-Based Multi-Agent Prompting Strategy for Text Classification 

**Title (ZH)**: 不只是演示，教给我原理：一种基于原理的多agent提示策略用于文本分类 

**Authors**: Peipei Wei, Dimitris Dimitriadis, Yan Xu, Mingwei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.07165)  

**Abstract**: We present PRINCIPLE-BASED PROMPTING, a simple but effective multi-agent prompting strategy for text classification. It first asks multiple LLM agents to independently generate candidate principles based on analysis of demonstration samples with or without labels, consolidates them into final principles via a finalizer agent, and then sends them to a classifier agent to perform downstream classification tasks. Extensive experiments on binary and multi-class classification datasets with different sizes of LLMs show that our approach not only achieves substantial performance gains (1.55% - 19.37%) over zero-shot prompting on macro-F1 score but also outperforms other strong baselines (CoT and stepback prompting). Principles generated by our approach help LLMs perform better on classification tasks than human crafted principles on two private datasets. Our multi-agent PRINCIPLE-BASED PROMPTING approach also shows on-par or better performance compared to demonstration-based few-shot prompting approaches, yet with substantially lower inference costs. Ablation studies show that label information and the multi-agent cooperative LLM framework play an important role in generating high-quality principles to facilitate downstream classification tasks. 

**Abstract (ZH)**: 我们提出了基于原则的提示策略（PRINCIPLE-BASED PROMPTING），这是一种简单而有效的多智能体提示策略，适用于文本分类任务。该方法首先让多个大型语言模型（LLM）智能体独立地基于示例样本的分析（带标签或不带标签）生成候选原则，然后通过一个最终处理智能体将这些原则整合为最终原则，并将最终原则发送给分类智能体以执行下游分类任务。在不同大小的LLM上进行的二分类和多分类数据集的广泛实验表明，我们的方法不仅在宏F1评分上优于零-shot提示（1.55% - 19.37%的显著性能提升），还优于其他强基线方法（即显性推理和反向提示）。通过我们方法生成的原则在两个私有数据集上帮助LLM在分类任务上表现优于手工设计的原则。此外，我们的多智能体PRINCIPLE-BASED PROMPTING方法在性能上与基于示例的少样本提示方法持平或更好，但推理成本显著降低。消融研究显示，标签信息和多智能体合作的LLM框架在生成高质量原则以促进下游分类任务方面发挥着重要作用。 

---
# Ask Patients with Patience: Enabling LLMs for Human-Centric Medical Dialogue with Grounded Reasoning 

**Title (ZH)**: 耐心问询患者：通过基于推理的支持来实现以人文本的医学对话的人工智能模型 

**Authors**: Jiayuan Zhu, Junde Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.07143)  

**Abstract**: Accurate and efficient diagnosis in online medical consultations remains a challenge for current large language models. These models often rely on single-turn interactions and lack the ability to refine their predictions through follow-up questions. Additionally, their responses frequently contain complex medical terminology, making them less accessible to non-medical users and creating barriers to effective communication. In this paper, we introduce Ask Patients with Patience (APP), the first multi-turn dialogue that enables LLMs to iteratively refine diagnoses based on grounded reasoning. By integrating medical guidelines and entropy minimization, APP improves both diagnostic accuracy and efficiency. Furthermore, it features human-centric communication that bridges the gap between user comprehension and medical terminology, significantly enhancing user accessibility and engagement. We evaluated APP using a subset of the ReMeDi dataset, comparing it with single-turn and traditional multi-turn LLM baselines. APP achieved higher similarity scores in diagnosis predictions, demonstrating better alignment with ground truth diagnoses. Entropy analysis showed that APP reduces diagnostic uncertainty more rapidly across iterations, increasing confidence in its predictions. APP also excels in user accessibility and empathy, further bridging the gap between complex medical language and user understanding. Code will be released at: this https URL. 

**Abstract (ZH)**: 当前的大语言模型在在线医疗咨询中实现准确且高效的诊断仍面临挑战。这些模型通常依赖单一回合的交互，缺乏通过后续问题来逐步细化预测的能力。此外，它们的回答经常包含复杂医学术语，这使得对非医疗用户来说不太易懂，从而成为有效沟通的障碍。在本文中，我们提出了“耐心询问患者”（Ask Patients with Patience, APP），这是第一个多回合对话系统，能够使大语言模型基于情境推理逐步细化预测。通过结合医学指南和最小熵原理，APP提高了诊断的准确性和效率。此外，APP还包括以用户体验为中心的沟通方式，这种沟通方式缩小了用户理解和医学术语之间的差距，显著提高了用户使用的易用性和参与度。我们使用ReMeDi数据集的一部分对APP进行了评估，并将其与单回合和传统的多回合大语言模型基准进行了比较。APP在诊断预测相似度方面得分更高，表明其预测更有助于与真实诊断对齐。熵分析显示，APP在每次迭代中更快地减少了诊断不确定性，增强了其预测的信心。此外，APP在用户易用性和同理心方面表现出色，进一步缩小了复杂医学语言与用户理解之间的差距。代码将发布于：this https URL。 

---
# Multi-turn Evaluation of Anthropomorphic Behaviours in Large Language Models 

**Title (ZH)**: 大型语言模型中类人行为的多轮评估 

**Authors**: Lujain Ibrahim, Canfer Akbulut, Rasmi Elasmar, Charvi Rastogi, Minsuk Kahng, Meredith Ringel Morris, Kevin R. McKee, Verena Rieser, Murray Shanahan, Laura Weidinger  

**Link**: [PDF](https://arxiv.org/pdf/2502.07077)  

**Abstract**: The tendency of users to anthropomorphise large language models (LLMs) is of growing interest to AI developers, researchers, and policy-makers. Here, we present a novel method for empirically evaluating anthropomorphic LLM behaviours in realistic and varied settings. Going beyond single-turn static benchmarks, we contribute three methodological advances in state-of-the-art (SOTA) LLM evaluation. First, we develop a multi-turn evaluation of 14 anthropomorphic behaviours. Second, we present a scalable, automated approach by employing simulations of user interactions. Third, we conduct an interactive, large-scale human subject study (N=1101) to validate that the model behaviours we measure predict real users' anthropomorphic perceptions. We find that all SOTA LLMs evaluated exhibit similar behaviours, characterised by relationship-building (e.g., empathy and validation) and first-person pronoun use, and that the majority of behaviours only first occur after multiple turns. Our work lays an empirical foundation for investigating how design choices influence anthropomorphic model behaviours and for progressing the ethical debate on the desirability of these behaviours. It also showcases the necessity of multi-turn evaluations for complex social phenomena in human-AI interaction. 

**Abstract (ZH)**: 用户倾向于将大型语言模型（LLMs）拟人化的趋势日益引起人工智能开发者、研究人员和政策制定者的关注。在此，我们提出了一种新颖的方法来实证评估拟人化LLM的行为，这些评估在不同的现实场景中进行。超越单一回合静态基准测试，我们在此贡献了最先进的评估（SOTA）方法的三项进步。首先，我们开发了一种多回合评估14种拟人化行为的方法。其次，我们通过运用用户交互的模拟，提出了一种可扩展且自动化的方案。第三，我们进行了一项交互式的大规模人类被试研究（N=1101），以验证我们测量的模型行为能够预测实际用户的拟人化感知。研究发现，在评估的所有SOTA LLMs中，它们都表现出类似的行为，这些行为特征主要包括关系构建（例如，同理心和验证）以及第一人称代词的使用，并且多数行为仅在多回合交互后出现。我们的工作奠定了一个实证基础，用于研究设计选择如何影响拟人化模型的行为，并促进有关这些行为的伦理讨论的进展。此外，我们的研究还突显了在人机交互中评估复杂社会现象的必要性，需采用多回合评估方式。 

---
# Specializing Large Language Models to Simulate Survey Response Distributions for Global Populations 

**Title (ZH)**: 将大型语言模型专门化以模拟全球人口的调查响应分布 

**Authors**: Yong Cao, Haijiang Liu, Arnav Arora, Isabelle Augenstein, Paul Röttger, Daniel Hershcovich  

**Link**: [PDF](https://arxiv.org/pdf/2502.07068)  

**Abstract**: Large-scale surveys are essential tools for informing social science research and policy, but running surveys is costly and time-intensive. If we could accurately simulate group-level survey results, this would therefore be very valuable to social science research. Prior work has explored the use of large language models (LLMs) for simulating human behaviors, mostly through prompting. In this paper, we are the first to specialize LLMs for the task of simulating survey response distributions. As a testbed, we use country-level results from two global cultural surveys. We devise a fine-tuning method based on first-token probabilities to minimize divergence between predicted and actual response distributions for a given question. Then, we show that this method substantially outperforms other methods and zero-shot classifiers, even on unseen questions, countries, and a completely unseen survey. While even our best models struggle with the task, especially on unseen questions, our results demonstrate the benefits of specialization for simulation, which may accelerate progress towards sufficiently accurate simulation in the future. 

**Abstract (ZH)**: 大规模调查是社会科学研究和政策制定的重要工具，但进行调查需要耗费大量的资金和时间。如果能够准确模拟群体级别的调查结果，这将对社会科学研究极具价值。以往的研究探索了使用大型语言模型（LLMs）模拟人类行为的方法，主要是通过提示来实现。在本文中，我们首次专门将LLMs用于模拟调查回应分布的任务。为测试这一方法，我们使用了两个全球文化调查的国家级结果。我们设计了一种基于初始词概率的微调方法，以最小化预测与实际回应分布之间的偏离程度，特别是针对特定问题。然后，我们展示了这种方法在其他方法和零样本分类器上取得了显著的优势，即使在未见过的问题、国家和全新的调查上也是如此。虽然我们最好的模型在任务上仍遇到困难，尤其是在未见过的问题上，但我们的结果表明专门化对于模拟的好处，这可能有助于未来实现更加准确的模拟。 

---
# Using Contextually Aligned Online Reviews to Measure LLMs' Performance Disparities Across Language Varieties 

**Title (ZH)**: 使用上下文对齐的在线评论来衡量LLM在不同语言变体中的性能差异 

**Authors**: Zixin Tang, Chieh-Yang Huang, Tsung-Chi Li, Ho Yim Sam Ng, Hen-Hsen Huang, Ting-Hao 'Kenneth' Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.07058)  

**Abstract**: A language can have different varieties. These varieties can affect the performance of natural language processing (NLP) models, including large language models (LLMs), which are often trained on data from widely spoken varieties. This paper introduces a novel and cost-effective approach to benchmark model performance across language varieties. We argue that international online review platforms, such as this http URL, can serve as effective data sources for constructing datasets that capture comments in different language varieties from similar real-world scenarios, like reviews for the same hotel with the same rating using the same language (e.g., Mandarin Chinese) but different language varieties (e.g., Taiwan Mandarin, Mainland Mandarin). To prove this concept, we constructed a contextually aligned dataset comprising reviews in Taiwan Mandarin and Mainland Mandarin and tested six LLMs in a sentiment analysis task. Our results show that LLMs consistently underperform in Taiwan Mandarin. 

**Abstract (ZH)**: 一种语言可以有不同的变体。这些变体可能会影响自然语言处理（NLP）模型的表现，包括大型语言模型（LLMs），这些模型通常是基于广泛使用的语言变体的数据进行训练的。本文介绍了一种新颖且成本效益高的方法，用于跨语言变体基准测试模型性能。我们认为，国际在线评论平台，如 TripAdvisor等，可以作为有效数据来源，用于构建捕捉不同语言变体但在相似现实场景中的评论数据集，例如使用相同语言（如普通话）对相同酒店给予相同评分的评论，但使用不同的语言变体（如台湾普通话、大陆普通话）。为了验证这一概念，我们构建了一个上下文对齐的数据集，包括台湾普通话和大陆普通话的评论，并在情感分析任务中测试了六种LLMs。我们的结果表明，LLMs在台湾普通话中的表现始终较差。 

---
# Demystifying Singular Defects in Large Language Models 

**Title (ZH)**: 揭开大型语言模型中独特缺陷的面纱 

**Authors**: Haoqi Wang, Tong Zhang, Mathieu Salzmann  

**Link**: [PDF](https://arxiv.org/pdf/2502.07004)  

**Abstract**: Large transformer models are known to produce high-norm tokens. In vision transformers (ViTs), such tokens have been mathematically modeled through the singular vectors of the linear approximations of layers. However, in large language models (LLMs), the underlying causes of high-norm tokens remain largely unexplored, and their different properties from those of ViTs require a new analysis framework. In this paper, we provide both theoretical insights and empirical validation across a range of recent models, leading to the following observations: i) The layer-wise singular direction predicts the abrupt explosion of token norms in LLMs. ii) The negative eigenvalues of a layer explain its sudden decay. iii) The computational pathways leading to high-norm tokens differ between initial and noninitial tokens. iv) High-norm tokens are triggered by the right leading singular vector of the matrix approximating the corresponding modules. We showcase two practical applications of these findings: the improvement of quantization schemes and the design of LLM signatures. Our findings not only advance the understanding of singular defects in LLMs but also open new avenues for their application. We expect that this work will stimulate further research into the internal mechanisms of LLMs and will therefore publicly release our code. 

**Abstract (ZH)**: 大型变压器模型已知会产生高范数的token。在视觉变压器（ViTs）中，这些token已经被通过层的线性近似奇异向量的数学模型进行了描述。然而，在大型语言模型（LLMs）中，导致高范数token的根本原因仍然很大程度上未被探究，且其与ViTs的不同的属性要求一个新的分析框架。本文中我们通过一系列最近模型提供了理论上的洞察和实证验证，得出以下观察结果：i) 层级奇异方向预测了LLMs中token范数的突然爆炸。ii) 层的负特征值解释了其突然衰减的原因。iii) 导致高范数token的计算路径在初始token和非初始token之间有所不同。iv) 高范数token由近似对应模块矩阵的右主奇异向量触发。我们展示了这些发现的两个实际应用：量化方案的改进和LLMs签名的设计。我们的发现不仅有助于理解LLMs中的奇异缺陷，还为它们的应用开启了新的途径。我们期待这项工作能激发更多关于LLMs内部机制的研究，因此我们将公开发布我们的代码。 

---
# Related Knowledge Perturbation Matters: Rethinking Multiple Pieces of Knowledge Editing in Same-Subject 

**Title (ZH)**: 相关的知识扰动事项：重新审视同一主题下的多知识点编辑 

**Authors**: Zenghao Duan, Wenbin Duan, Zhiyi Yin, Yinghan Shen, Shaoling Jing, Jie Zhang, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.06868)  

**Abstract**: Knowledge editing has become a promising approach for efficiently and precisely updating knowledge embedded in large language models (LLMs). In this work, we focus on Same-Subject Editing, which involves modifying multiple attributes of a single entity to ensure comprehensive and consistent updates to entity-centric knowledge. Through preliminary observation, we identify a significant challenge: Current state-of-the-art editing methods struggle when tasked with editing multiple related knowledge pieces for the same subject. To address the lack of relevant editing data for identical subjects in traditional benchmarks, we introduce the $\text{S}^2\text{RKE}$(Same-Subject Related Knowledge Editing) benchmark. Our extensive experiments reveal that only mainstream locate-then-edit methods, such as ROME and MEMIT, exhibit "related knowledge perturbation," where subsequent edits interfere with earlier ones. Further analysis reveals that these methods over-rely on subject information, neglecting other critical factors, resulting in reduced editing effectiveness. 

**Abstract (ZH)**: 知识编辑已成为高效且精确更新大型语言模型（LLMs）中嵌入知识的一种有前途的方法。在本文中，我们重点关注“同主题编辑（Same-Subject Editing）”，其涉及修改单个实体的多个属性，以确保实体为中心的知识的全面和一致性更新。通过初步观察，我们发现一个重大挑战：当前最先进的编辑方法在处理同一主题的多个相关知识片段的编辑任务时表现不佳。为了解决传统基准中对于相同主题的相关编辑数据缺乏的问题，我们引入了$\text{S}^2\text{RKE}$(Same-Subject Related Knowledge Editing)基准。我们的大量实验表明，只有主流的“查找-编辑”方法（如ROME和MEMIT）表现出“相关知识扰动”，即后续编辑干扰了先前的编辑。进一步分析表明，这些方法过度依赖主题信息，忽视了其他关键因素，从而导致编辑效果降低。 

---
# Self-Supervised Prompt Optimization 

**Title (ZH)**: 自主监督提示优化 

**Authors**: Jinyu Xiang, Jiayi Zhang, Zhaoyang Yu, Fengwei Teng, Jinhao Tu, Xinbing Liang, Sirui Hong, Chenglin Wu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.06855)  

**Abstract**: Well-designed prompts are crucial for enhancing Large language models' (LLMs) reasoning capabilities while aligning their outputs with task requirements across diverse domains. However, manually designed prompts require expertise and iterative experimentation. While existing prompt optimization methods aim to automate this process, they rely heavily on external references such as ground truth or by humans, limiting their applicability in real-world scenarios where such data is unavailable or costly to obtain. To address this, we propose Self-Supervised Prompt Optimization (SPO), a cost-efficient framework that discovers effective prompts for both closed and open-ended tasks without requiring external reference. Motivated by the observations that prompt quality manifests directly in LLM outputs and LLMs can effectively assess adherence to task requirements, we derive evaluation and optimization signals purely from output comparisons. Specifically, SPO selects superior prompts through pairwise output comparisons evaluated by an LLM evaluator, followed by an LLM optimizer that aligns outputs with task requirements. Extensive experiments demonstrate that SPO outperforms state-of-the-art prompt optimization methods, achieving comparable or superior results with significantly lower costs (e.g., 1.1% to 5.6% of existing methods) and fewer samples (e.g., three samples). The code is available at this https URL. 

**Abstract (ZH)**: 精心设计的提示对于增强大型语言模型（LLMs）的推理能力并使其输出符合跨多种领域的任务要求至关重要。然而，手动设计提示需要专业知识和迭代实验。虽然现有提示优化方法力求自动完成这一过程，但它们严重依赖外部参考，如地面真相或人工数据，这限制了它们在获取此类数据不可用或成本高昂的实际场景中的应用。为解决这一问题，我们提出了一种成本效益高的自监督提示优化（SPO）框架，该框架可以在无需外部参考的情况下发现适用于闭合和开放任务的有效提示。受实观察的启发，提示质量直接反映在LLM输出中，且LLMs能够有效地评估对任务要求的符合程度，我们从输出比较中纯粹提取评估和优化信号。具体而言，SPO通过LLM评估器对成对输出进行比较选择优越的提示，并通过LLM优化器将输出与任务要求对齐。广泛实验表明，SPO优于现有的最佳提示优化方法，能够在显著降低成本（例如，现有方法的1.1%到5.6%）和较少样本（例如，三个样本）的情况下获得相当或更优的结果。代码可在此处访问：[该网址]。 

---
# Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models 

**Title (ZH)**: 面向多模态大规模语言模型的零样本异常检测与推理 

**Authors**: Jiacong Xu, Shao-Yuan Lo, Bardia Safaei, Vishal M. Patel, Isht Dwivedi  

**Link**: [PDF](https://arxiv.org/pdf/2502.07601)  

**Abstract**: Zero-Shot Anomaly Detection (ZSAD) is an emerging AD paradigm. Unlike the traditional unsupervised AD setting that requires a large number of normal samples to train a model, ZSAD is more practical for handling data-restricted real-world scenarios. Recently, Multimodal Large Language Models (MLLMs) have shown revolutionary reasoning capabilities in various vision tasks. However, the reasoning of image abnormalities remains underexplored due to the lack of corresponding datasets and benchmarks. To facilitate research in AD & reasoning, we establish the first visual instruction tuning dataset, Anomaly-Instruct-125k, and the evaluation benchmark, VisA-D&R. Through investigation with our benchmark, we reveal that current MLLMs like GPT-4o cannot accurately detect and describe fine-grained anomalous details in images. To address this, we propose Anomaly-OneVision (Anomaly-OV), the first specialist visual assistant for ZSAD and reasoning. Inspired by human behavior in visual inspection, Anomaly-OV leverages a Look-Twice Feature Matching (LTFM) mechanism to adaptively select and emphasize abnormal visual tokens. Extensive experiments demonstrate that Anomaly-OV achieves significant improvements over advanced generalist models in both detection and reasoning. Extensions to medical and 3D AD are provided for future study. The link to our project page: this https URL 

**Abstract (ZH)**: 零样本异常检测（ZSAD）是一种新兴的异常检测（AD）范式。与传统无监督AD设置需要大量正常样本来训练模型不同，ZSAD在处理数据受限的实际场景时更为实用。最近，多模态大语言模型（MLLMs）在各种视觉任务中展现出革命性的推理能力。然而，由于缺乏相应的数据集和基准，图像异常的推理仍然未被充分探索。为促进AD与推理的研究，我们建立了首个视觉指令调优数据集Anomaly-Instruct-125k及评估基准VisA-D&R。通过我们的基准评估，我们揭示了当前的MLLMs，如GPT-4o，无法准确地检测和描述图像中的细微异常细节。为了应对这一挑战，我们提出了首个专门针对ZSAD和推理的视觉助手Anomaly-OneVision（Anomaly-OV）。Anomaly-OV借鉴了人类视觉检查行为，利用“双查看特征匹配”（LTFM）机制来自适应地选择和强调异常视觉特征。广泛的实验表明，Anomaly-OV在检测和推理两个方面都显著优于先进的通用模型。我们还为未来的研究提供了医疗和三维AD的应用扩展。我们的项目页面链接：[这个链接](https://your-project-page-url.com) 

---
# DrugImproverGPT: A Large Language Model for Drug Optimization with Fine-Tuning via Structured Policy Optimization 

**Title (ZH)**: DrugImproverGPT：一种通过结构化策略优化进行微调的大语言模型用于药物优化 

**Authors**: Xuefeng Liu, Songhao Jiang, Siyu Chen, Zhuoran Yang, Yuxin Chen, Ian Foster, Rick Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2502.07237)  

**Abstract**: Finetuning a Large Language Model (LLM) is crucial for generating results towards specific objectives. This research delves into the realm of drug optimization and introduce a novel reinforcement learning algorithm to finetune a drug optimization LLM-based generative model, enhancing the original drug across target objectives, while retains the beneficial chemical properties of the original drug. This work is comprised of two primary components: (1) DrugImprover: A framework tailored for improving robustness and efficiency in drug optimization. It includes a LLM designed for drug optimization and a novel Structured Policy Optimization (SPO) algorithm, which is theoretically grounded. This algorithm offers a unique perspective for fine-tuning the LLM-based generative model by aligning the improvement of the generated molecule with the input molecule under desired objectives. (2) A dataset of 1 million compounds, each with OEDOCK docking scores on 5 human proteins associated with cancer cells and 24 binding sites from SARS-CoV-2 virus. We conduct a comprehensive evaluation of SPO and demonstrate its effectiveness in improving the original drug across target properties. Our code and dataset will be publicly available at: this https URL. 

**Abstract (ZH)**: 微调大型语言模型（LLM）对于生成特定目标的成果至关重要。本研究深入探讨了药物优化领域，并提出了一种新颖的强化学习算法来微调基于药物优化的LLM生成模型，该模型通过保留原始药物的有利化学特性，同时增强目标药物性能。本研究主要包括两个主要组成部分：(1) DrugImprover：一个专门用于提高药物优化稳健性和效率的框架。该框架包括一个用于药物优化的LLM，以及一个新颖的结构化策略优化（SPO）算法，该算法具有深厚的理论基础。该算法通过将生成分子的改进与输入分子在期望目标下的性能对齐，为微调基于LLM的生成模型提供了独特的视角。(2) 包含100万个化合物的数据库，每个化合物都有与癌症细胞相关的5个人体蛋白质和SARS-CoV-2病毒的24个结合位点的OEDOCK对接评分。我们对SPO进行了全面评估，并证明了其在改善目标药物性能方面的有效性。我们的代码和数据库将在以下公开访问：this https URL。 

---
