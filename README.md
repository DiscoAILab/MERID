![MERID](assets/merid_banner.png)

**MERID: Multimodal Exploration via Recursive Self-Improvement Agents for Major Depression Analysis**

MERID builds multimodal depression-detection pipelines through an experience-driven recursive
self-improvement (RSI) loop. Each round proposes pipeline modifications, verifies them under a fixed
development protocol, and carries the verified pipeline and the accumulated experience into the
next round.

## Highlights

- **Grounded State Construction (GSC).** The agent derives a data adapter for each corpus and
  verifies it with execution checks and recipe probes, so every feature recipe sees the same
  subjects, records and targets.
- **Coupled Pipeline Exploration (CPE).** Representation, fusion and predictor choices are searched
  jointly as complete pipelines under one shared development protocol.
- **Evidence-Guided Evolution (EGE).** Planner, Engineer and Reviewer agents analyze the measured
  outcomes, the Chair turns their reports into revision requests, and a revision is inherited when
  its gain exceeds a subject-bootstrap margin.

## How it works

| Step | What happens | Code |
|---|---|---|
| 1. Inspect | The agent profiles the corpus: files, modalities, subjects and label provenance. | `adapter_agent.inspect_dataset`, `label_provenance.py` |
| 2. Propose | The LLM proposes the feature-recipe space, the predictor space and, where the benchmark fixes none, the metric. | `run_method.py`, `run_corpus.py`, `retrieval.py`, `operation.py` |
| 3. Ground (GSC) | The agent writes the data adapter; execution checks and recipe probes verify it, and failed checks trigger repair. | `AdapterAgent.create_adapter` |
| 4. Explore (CPE) | Recipe × predictor pairs are scored on fixed subject-disjoint folds or the official development split. | `DevelopmentSearch.evaluate` |
| 5. Evolve (EGE) | The review roles report to the Chair, who decides ACCEPT or REVISE. Revisions are scored and retained past the bootstrap margin, and the research memory is updated. | `retrospective.hold_review`, `development_search.py`, `research_memory.py` |
| 6. Inherit | The retained pipeline, candidate space, history and memory carry into the next round. | `development_agent.run_development` |
| 7. Test once | The development-selected deployment is scored once on the held-out test set. | `corpus_final_eval.py`, `mpdd2025_final_eval.py`, `mpdd2026_final_eval.py` |

MPDD-2026 runs the search-and-review loop over its binary/PHQ and hierarchical ternary head
families (`run_agent.py`, `benchmarks/mpdd2026/joint_search.py`).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── assets/
│   └── merid_banner.png
└── merid/
    ├── run_corpus.py                        entry point: E-DAIC, Mental Health, Kaggle
    ├── run_method.py                        entry point: MPDD-2025
    ├── run_agent.py                         entry point: MPDD-2026
    ├── core/
    │   ├── development_agent.py             self-improvement loop: search, review rounds, inheritance
    │   ├── grounded_state_construction/     GSC
    │   │   ├── adapter_agent.py             adapter derivation, execution checks, recipe probes, repair
    │   │   ├── adapter_cache.py             cache of deterministic adapter outputs
    │   │   ├── adapter_source_evidence.py   adapter source as review evidence
    │   │   └── label_provenance.py          target-defining column check
    │   ├── coupled_pipeline_exploration/    CPE
    │   │   ├── development_search.py        joint recipe × predictor search
    │   │   ├── registry.py                  method catalog
    │   │   └── operation.py                 generated estimators for the catalog
    │   └── evidence_guided_evolution/       EGE
    │       ├── retrospective.py             Planner, Engineer, Reviewer and Chair
    │       ├── research_memory.py           lessons distilled after each review round
    │       ├── reliability.py               uncertainty-aware verification
    │       └── development_diagnostics.py   selection evidence for the review
    ├── llm/                                 proposer and web-search grounding
    ├── utils/                               run trajectory and progress reporting
    ├── benchmarks/
    │   ├── mpdd2025/                        development entry, grouped decoding, estimator factories
    │   └── mpdd2026/                        data, heads, fold protocol, metrics, head search
    └── evaluation/                          final evaluation on the held-out test sets
```

Each module opens with a one-line summary of its role. Paths specific to our computing environment
appear as placeholders such as `<SPLIT_ROOT>`.

## Prompts

The prompts reproduced in the appendix are defined in these modules (paths relative to `merid/`):

| Prompt | Definition |
|---|---|
| Benchmark metric | `run_method.py` `EVAL_METRIC_SYSTEM` |
| Feature-recipe space | `run_corpus.py` `RECIPE_SYSTEM`; `run_method.py` `RECIPE_SPACE_SYSTEM` |
| Classifier and regressor spaces | `run_method.py` `CLF_SPACE_SYSTEM`, `REG_SPACE_SYSTEM` |
| Model-code generation | `run_method.py` `CODEGEN_SYSTEM` |
| Data adapter derivation and repair | `core/grounded_state_construction/adapter_agent.py` `ADAPTER_SYSTEM` |
| Method selection | `core/coupled_pipeline_exploration/registry.py` `select_method` |
| MPDD-2026 head proposer | `benchmarks/mpdd2026/joint_search.py` `HeadProposer.propose` |
| Planner, Engineer, Reviewer, Chair | `core/evidence_guided_evolution/retrospective.py` `ROLE_SYSTEM`, `CHAIR_SYSTEM`, `REVISION_REQUIRED`, `_INVESTIGATE` |
| Distill and Reconcile | `core/evidence_guided_evolution/research_memory.py` `DISTILL_SYSTEM`, `RECONCILE_SYSTEM` |
| Web-search clause | `llm/retrieval.py` `OPTIONAL_SEARCH_CLAUSE` |
