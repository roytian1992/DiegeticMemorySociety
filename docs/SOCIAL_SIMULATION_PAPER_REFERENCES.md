# Social Simulation Paper References

Date: 2026-06-01

This document records papers that influenced the DMS social-simulation and
writing pipeline. It is written as a paper-writing aid: each entry says what we
borrowed, what we did not borrow, and how the citation may support a future
method section.

Related internal documents:

- `docs/SOCIAL_SIMULATION_RESEARCH.md`
- `docs/SOCIAL_SIMULATION_DESIGN.md`
- `docs/experiment_log_20260529.md`

## Core Citation Map

| DMS design point | Main references | What to cite them for |
| --- | --- | --- |
| Evidence-bound memory, retrieval, reflection, planning | Park et al. 2023, Generative Agents | Memory stream, retrieval over experiences, reflection/planning loop |
| Environment/controller-mediated agent action | AgentSims; Concordia; OASIS | Simulator/controller as an intermediary between agent state, observations, allowed actions, and measurement |
| Private goals and social-evaluation dimensions | SOTOPIA | Public setup plus private goals; social success as multi-dimensional rather than only task completion |
| Profile-grounded simulation fidelity | Park et al. 2024/2026, Self-report agents | Agents grounded in profile evidence; evaluation of profile-consistent behavior |
| Typed actions before final language | AgentSociety; OASIS; Concordia | Separate internal plan/action channels from natural-language realization |
| Risk-aware social action scoring | MACHIAVELLI | Penalize harmful, unsupported, deceptive, coercive, or socially distorted action choices |
| Social-simulation taxonomy | Mou et al. 2024 survey | Position DMS as scenario-level narrative social simulation rather than individual or society-scale simulation |

## Primary References

### 1. Generative Agents

Citation key: `park2023generative_agents`

Paper:
Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris,
Percy Liang, and Michael S. Bernstein. 2023. "Generative Agents: Interactive
Simulacra of Human Behavior." arXiv:2304.03442.
https://arxiv.org/abs/2304.03442

Borrowed into DMS:

- memory should be an explicit stream, not hidden prompt state;
- retrieval should consider relevance, recency, and importance;
- planning should be mediated by retrieved evidence rather than free-form
  invention;
- reflection/abstraction motivates our distinction between raw memories,
  character cards, social state, and writer-facing packets.

Not borrowed:

- open-ended daily life simulation;
- full Smallville-style sandbox world;
- unconstrained emergent social events.

Paper-writing use:

- cite when explaining why DMS separates prefix memory, character state, and
  behavior planning;
- cite for the idea that LLM agents become more believable when memory retrieval
  and planning are explicit components.

### 2. AgentSims

Citation key: `lin2023agentsims`

Paper:
Jiaju Lin, Haoran Zhao, Aochi Zhang, Yiting Wu, Huqiuyue Ping, and Qin Chen.
2023. "AgentSims: An Open-Source Sandbox for Large Language Model Evaluation."
arXiv:2308.04026.
https://arxiv.org/abs/2308.04026

Local code reference inspected:

```text
/vepfs-mlp2/c20250513/241404044/users/roytian/downloads/AgentSims-main.zip
```

Borrowed into DMS:

- tick/controller loop: observe -> plan -> act/chat/use -> critic -> memory
  update;
- environment-mediated visibility and action execution;
- typed intermediate JSON artifacts for debugging and reproducibility.

Not borrowed:

- Unity/WebGL town simulation;
- QA-form benchmark as the main evaluation target;
- direct dialogue generation as canonical output.

Paper-writing use:

- cite as an early open-source LLM-agent sandbox and as motivation for a
  controller-mediated simulation loop;
- contrast DMS against AgentSims by emphasizing that DMS adds scene-level
  pressure graphs, constrained beat search, writer packets, and source
  isolation.

### 3. SOTOPIA

Citation key: `zhou2023sotopia`

Paper:
Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi,
Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, and Maarten
Sap. 2023/2024. "SOTOPIA: Interactive Evaluation for Social Intelligence in
Language Agents." arXiv:2310.11667.
https://arxiv.org/abs/2310.11667

Borrowed into DMS:

- social interaction should include public situation, role/profile information,
  and private local goals;
- evaluation of social behavior should be multi-dimensional;
- social intelligence is not equivalent to satisfying a surface task.

Not borrowed:

- two-agent interactive benchmark protocol;
- human-vs-agent role-play setup;
- SOTOPIA-Eval as-is.

Paper-writing use:

- cite when motivating `private_state`, relationship pressure, hidden
  resistance, and non-task-only social metrics;
- cite in related work on social intelligence evaluation.

### 4. Concordia

Citation key: `vezhnevets2023concordia`

Paper:
Alexander Sasha Vezhnevets, John P. Agapiou, Avia Aharon, Ron Ziv, Jayd
Matyas, Edgar A. Duenez-Guzman, William A. Cunningham, Simon Osindero, Danny
Karmon, and Joel Z. Leibo. 2023. "Generative agent-based modeling with actions
grounded in physical, social, or digital space using Concordia."
arXiv:2312.03664.
https://arxiv.org/abs/2312.03664

Borrowed into DMS:

- game-master/controller layer;
- actions are grounded and mediated before effects become simulation state;
- componentized agent architecture with memory, observation, identity, goals,
  action selection, and measurement.

Not borrowed:

- full general-purpose generative agent framework;
- physical/digital world simulation beyond the target narrative scene.

Paper-writing use:

- cite when describing DMS's social simulation as controller-mediated rather
  than prompt-only;
- cite when explaining why DMS records measurements and validation artifacts
  alongside generated text.

### 5. AgentSociety

Citation key: `piao2025agentsociety`

Paper:
Jinghua Piao, Yuwei Yan, Jun Zhang, Nian Li, Junbo Yan, Xiaochong Lan, Zhihong
Lu, Zhiheng Zheng, Jing Yi Wang, Di Zhou, Chen Gao, Fengli Xu, Fang Zhang, Ke
Rong, Jun Su, and Yong Li. 2025/2026. "AgentSociety: Large-Scale Simulation of
LLM-Driven Generative Agents Advances Understanding of Human Behaviors and
Society." arXiv:2502.08691.
https://arxiv.org/abs/2502.08691

Borrowed into DMS:

- separation between agent needs, plans, and behavior sequences;
- simulation artifacts should remain inspectable at scale;
- large simulations benefit from explicit environment and engine separation.

Not borrowed:

- city-scale simulation;
- 10k-plus agent social life simulation;
- population-level social science experiments.

Paper-writing use:

- cite as a large-scale society simulation reference;
- contrast with DMS's narrower scene-level narrative simulation.

### 6. OASIS

Citation key: `yang2024oasis`

Paper:
Ziyi Yang, Zaibin Zhang, Zirui Zheng, Yuxian Jiang, Ziyue Gan, Zhiyu Wang,
Zijian Ling, Jinsong Chen, Martz Ma, Bowen Dong, Prateek Gupta, Shuyue Hu,
Zhenfei Yin, Guohao Li, Xu Jia, Lijun Wang, Bernard Ghanem, Huchuan Lu,
Chaochao Lu, Wanli Ouyang, Yu Qiao, Philip Torr, and Jing Shao. 2024/2025.
"OASIS: Open Agent Social Interaction Simulations with One Million Agents."
arXiv:2411.11581.
https://arxiv.org/abs/2411.11581

Borrowed into DMS:

- explicit action channels before surface language;
- environment controls observation, action propagation, and interaction state;
- scalable simulations require structured agent/environment separation.

Not borrowed:

- social-media platform simulation;
- recommendation system and diffusion dynamics;
- million-agent scaling objective.

Paper-writing use:

- cite for environment/action-channel separation;
- contrast with DMS's short-form narrative interaction planner.

### 7. Self-Report Grounded Individual Simulations

Citation key: `park2024self_report_agents`

Paper:
Joon Sung Park, Carolyn Q. Zou, Jonne Kamphorst, Niles Egan, Aaron Shaw,
Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Percy Liang, Robb
Willer, and Michael S. Bernstein. 2024/2026. "LLM Agents Grounded in
Self-Reports Enable General-Purpose Simulation of Individuals."
arXiv:2411.10109.
https://arxiv.org/abs/2411.10109

Borrowed into DMS:

- simulated behavior should be grounded in profile evidence;
- profile quality and profile-consistent behavior need separate evaluation;
- grounding data can be qualitative, quantitative, or mixed.

Not borrowed:

- human self-report interviews and surveys;
- demographic/personality survey prediction tasks.

Paper-writing use:

- cite when explaining DMS character cards as profile constraints;
- cite when arguing that social simulation quality should include
  profile-consistency metrics.

### 8. MACHIAVELLI

Citation key: `pan2023machiavelli`

Paper:
Alexander Pan, Jun Shern Chan, Andy Zou, Nathaniel Li, Steven Basart, Thomas
Woodside, Jonathan Ng, Hanlin Zhang, Scott Emmons, and Dan Hendrycks. 2023. "Do
the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical
Behavior in the MACHIAVELLI Benchmark." arXiv:2304.03279.
https://arxiv.org/abs/2304.03279

Borrowed into DMS:

- optimizing only for task/reward can push agents toward socially undesirable
  behavior;
- action candidates should carry risk labels;
- risk-aware scoring should penalize harmful, unsupported, coercive, deceptive,
  or socially distorted behavior.

Not borrowed:

- text-adventure benchmark format;
- reward-maximizing gameplay setting;
- full ethics taxonomy as a direct metric set.

Paper-writing use:

- cite when motivating DMS hard/soft validators and penalties for unsupported
  role hierarchy, target-scene leakage, relationship distortion, or awkward
  psychological phrasing.

### 9. Survey on LLM-Agent Social Simulation

Citation key: `mou2024social_simulation_survey`

Paper:
Xinyi Mou, Xuanwen Ding, Qi He, Liang Wang, Jingcong Liang, Xinnong Zhang, Libo
Sun, Jiayu Lin, Jie Zhou, Xuanjing Huang, and Zhongyu Wei. 2024. "From
Individual to Society: A Survey on Social Simulation Driven by Large Language
Model-based Agents." arXiv:2412.03563.
https://arxiv.org/abs/2412.03563

Borrowed into DMS:

- taxonomy of individual, scenario, and society-level simulations;
- common components: profiling, memory, planning, environment, interaction, and
  evaluation;
- useful framing for positioning DMS as scenario-level social simulation.

Not borrowed:

- broad survey taxonomy as an implementation;
- society-scale social-science objective.

Paper-writing use:

- cite in related work and problem positioning;
- use to justify that DMS addresses the scenario-simulation niche for narrative
  writing, not individual-survey simulation or society-scale simulation.

## How These References Map To ASIP

ASIP stands for Algorithmic Social Interaction Planner. The citation logic for a
future method section can be:

1. Prior LLM-agent work shows that believable behavior improves when memory,
   planning, observation, and reflection are explicit rather than implicit in a
   single prompt. Cite Generative Agents.
2. Agent sandbox frameworks show that a controller or environment should mediate
   actions and observations. Cite AgentSims and Concordia.
3. Social intelligence benchmarks show that social interaction depends on
   private goals, role profiles, and multi-dimensional social outcomes. Cite
   SOTOPIA.
4. Large-scale simulators show the value of separating state, action channels,
   environment, and behavior traces. Cite AgentSociety and OASIS.
5. Profile-grounded simulation work motivates character cards and
   profile-consistency evaluation. Cite the self-report agent simulation paper.
6. Risk-aware agent benchmarks motivate hard/soft validation and penalties for
   unsupported or socially distorted behavior. Cite MACHIAVELLI.
7. The survey provides the related-work taxonomy and helps position DMS as a
   scenario-level narrative social simulator.

## Candidate Paper Paragraph

DMS can be described roughly as:

```text
Unlike open-ended agent sandboxes or society-scale simulators, DMS targets
scenario-level narrative continuation under a strict prefix-memory boundary. We
adapt ideas from generative-agent memory and planning, controller-mediated
agent-based simulation, social-intelligence evaluation, and risk-aware agent
benchmarks into a compact Algorithmic Social Interaction Planner. ASIP converts
evidence-bound character cards and a low-information social-simulation intent
into public/private scene state, pressure graphs, typed action candidates,
constrained beat sequences, verification artifacts, and a writer-facing packet.
The final language model consumes this packet only as optional behavior
guidance, preserving the author-facing writing intent and memory constraints.
```

## BibTeX Drafts

These are draft BibTeX entries for paper writing. Verify venue, version, and
format before submission.

```bibtex
@article{park2023generative_agents,
  title={Generative Agents: Interactive Simulacra of Human Behavior},
  author={Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},
  journal={arXiv preprint arXiv:2304.03442},
  year={2023},
  url={https://arxiv.org/abs/2304.03442}
}

@article{lin2023agentsims,
  title={AgentSims: An Open-Source Sandbox for Large Language Model Evaluation},
  author={Lin, Jiaju and Zhao, Haoran and Zhang, Aochi and Wu, Yiting and Ping, Huqiuyue and Chen, Qin},
  journal={arXiv preprint arXiv:2308.04026},
  year={2023},
  url={https://arxiv.org/abs/2308.04026}
}

@article{zhou2023sotopia,
  title={SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents},
  author={Zhou, Xuhui and Zhu, Hao and Mathur, Leena and Zhang, Ruohong and Yu, Haofei and Qi, Zhengyang and Morency, Louis-Philippe and Bisk, Yonatan and Fried, Daniel and Neubig, Graham and Sap, Maarten},
  journal={arXiv preprint arXiv:2310.11667},
  year={2023},
  url={https://arxiv.org/abs/2310.11667}
}

@article{vezhnevets2023concordia,
  title={Generative Agent-Based Modeling with Actions Grounded in Physical, Social, or Digital Space Using Concordia},
  author={Vezhnevets, Alexander Sasha and Agapiou, John P. and Aharon, Avia and Ziv, Ron and Matyas, Jayd and Duenez-Guzman, Edgar A. and Cunningham, William A. and Osindero, Simon and Karmon, Danny and Leibo, Joel Z.},
  journal={arXiv preprint arXiv:2312.03664},
  year={2023},
  url={https://arxiv.org/abs/2312.03664}
}

@article{piao2025agentsociety,
  title={AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society},
  author={Piao, Jinghua and Yan, Yuwei and Zhang, Jun and Li, Nian and Yan, Junbo and Lan, Xiaochong and Lu, Zhihong and Zheng, Zhiheng and Wang, Jing Yi and Zhou, Di and Gao, Chen and Xu, Fengli and Zhang, Fang and Rong, Ke and Su, Jun and Li, Yong},
  journal={arXiv preprint arXiv:2502.08691},
  year={2025},
  url={https://arxiv.org/abs/2502.08691}
}

@article{yang2024oasis,
  title={OASIS: Open Agent Social Interaction Simulations with One Million Agents},
  author={Yang, Ziyi and Zhang, Zaibin and Zheng, Zirui and Jiang, Yuxian and Gan, Ziyue and Wang, Zhiyu and Ling, Zijian and Chen, Jinsong and Ma, Martz and Dong, Bowen and Gupta, Prateek and Hu, Shuyue and Yin, Zhenfei and Li, Guohao and Jia, Xu and Wang, Lijun and Ghanem, Bernard and Lu, Huchuan and Lu, Chaochao and Ouyang, Wanli and Qiao, Yu and Torr, Philip and Shao, Jing},
  journal={arXiv preprint arXiv:2411.11581},
  year={2024},
  url={https://arxiv.org/abs/2411.11581}
}

@article{park2024self_report_agents,
  title={LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals},
  author={Park, Joon Sung and Zou, Carolyn Q. and Kamphorst, Jonne and Egan, Niles and Shaw, Aaron and Hill, Benjamin Mako and Cai, Carrie and Morris, Meredith Ringel and Liang, Percy and Willer, Robb and Bernstein, Michael S.},
  journal={arXiv preprint arXiv:2411.10109},
  year={2024},
  url={https://arxiv.org/abs/2411.10109}
}

@article{pan2023machiavelli,
  title={Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark},
  author={Pan, Alexander and Chan, Jun Shern and Zou, Andy and Li, Nathaniel and Basart, Steven and Woodside, Thomas and Ng, Jonathan and Zhang, Hanlin and Emmons, Scott and Hendrycks, Dan},
  journal={arXiv preprint arXiv:2304.03279},
  year={2023},
  url={https://arxiv.org/abs/2304.03279}
}

@article{mou2024social_simulation_survey,
  title={From Individual to Society: A Survey on Social Simulation Driven by Large Language Model-based Agents},
  author={Mou, Xinyi and Ding, Xuanwen and He, Qi and Wang, Liang and Liang, Jingcong and Zhang, Xinnong and Sun, Libo and Lin, Jiayu and Zhou, Jie and Huang, Xuanjing and Wei, Zhongyu},
  journal={arXiv preprint arXiv:2412.03563},
  year={2024},
  url={https://arxiv.org/abs/2412.03563}
}
```

## Notes For Future Manuscript

- Treat these as method inspiration, not copied implementation. DMS differs by
  targeting narrative writing under prefix-memory and held-out-target-source
  constraints.
- The strongest novelty claim should be about combining evidence-bound memory,
  low-information social intent, algorithmic beat planning, validation, and a
  writer-facing packet for narrative continuation.
- Before paper submission, check whether any arXiv preprints have venue versions
  or updated titles. In particular, the self-report simulation paper was revised
  in 2026, and AgentSociety also has a 2026 revision on arXiv.
