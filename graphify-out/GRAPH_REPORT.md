# Graph Report - new_psi_fl  (2026-09-25)

## Corpus Check
- Corpus is ~7,745 words - fits in a single context window. You may not need a graph.

## Summary
- 243 nodes · 464 edges · 18 communities (10 shown, 8 thin omitted)
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 103 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Core Networking
- Model Aggregation
- CLI Config
- PACFL Clustering
- Main & Mapping Inputs
- Evaluation Metrics
- FL Client
- Smoke Testing
- Legacy Execution Tests
- Plans and Methods
- Integration Tests
- Initialization
- By Class Mapping
- Accuracy Metrics
- Agent Rules
- Agent Workflows
- GeFL Aggregation
- Old Acc Metric

## God Nodes (most connected - your core abstractions)
1. `execute_pair()` - 29 edges
2. `IntegrationTests` - 26 edges
3. `Client` - 20 edges
4. `Server` - 18 edges
5. `run_cli()` - 16 edges
6. `run()` - 15 edges
7. `GANState` - 14 edges
8. `ClientUpdate` - 13 edges
9. `ProtocolTests` - 12 edges
10. `ByClassMapping` - 11 edges

## Surprising Connections (you probably didn't know these)
- `image_bi` --semantically_similar_to--> `image_bi`  [INFERRED] [semantically similar]
  PLAN.md → README.md
- `GeFL` --semantically_similar_to--> `GeFL_gan_pacfl_iid`  [INFERRED] [semantically similar]
  PLAN.md → TRACE_REPORT.md
- `image_bi` --semantically_similar_to--> `image_bi`  [INFERRED] [semantically similar]
  TRACE_REPORT.md → README.md
- `IntegrationTests` --uses--> `GeFLAggregation`  [INFERRED]
  tests/test_integration.py → aggregation.py
- `execute_pair()` --uses--> `GeFLAggregation`  [INFERRED]
  tests/trace_legacy.py → aggregation.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Accuracy Evaluation Metrics** — readme_old_acc, readme_ground_truth_acc, readme_pure_semantic_class_only [EXTRACTED 1.00]

## Communities (18 total, 8 thin omitted)

### Community 0 - "Core Networking"
Cohesion: 0.08
Nodes (31): contextlib, csv, io, revised_protocol_aggregation, revised_protocol_client, revised_protocol_clustering, revised_protocol_evaluation, revised_protocol_main (+23 more)

### Community 1 - "Model Aggregation"
Cohesion: 0.10
Nodes (24): GeFLAggregation, Sample-weighted GeFL aggregation of GANs within PACFL groups., weighted_average(), Any, Independent GeFL participant. All incoming/outgoing state is copied., AggregationStrategy, ClientUpdate, clone_state() (+16 more)

### Community 2 - "CLI Config"
Cohesion: 0.08
Nodes (20): argparse, datetime, json, materialize_generators(), RunConfig, ImageBiMapping, Baseline image-bi entropy filtering + bidirectional cycle consistency., pathlib (+12 more)

### Community 3 - "PACFL Clustering"
Cohesion: 0.12
Nodes (9): PACFL, mapping_metrics(), Cross-group label-pair metrics; retain the legacy balanced accuracy/MCC. With…, ByClassMapping, RelationTable, Oracle: explicit local-index → semantic-class metadata, including permutations.…, noisy_mapping(), ProtocolTests (+1 more)

### Community 4 - "Main & Mapping Inputs"
Cohesion: 0.14
Nodes (14): MappingInputs, RelationTable, hashlib, main(), Composition and orchestration: PACFL → local GeFL → mapping → global…, label_spaces: client ID → ordered canonical semantic names. mapping_strategy is…, run(), seed_stage() (+6 more)

### Community 5 - "Evaluation Metrics"
Cohesion: 0.12
Nodes (10): evaluate_global(), old_acc(), Independent ground-truth relation and global-classifier evaluation., Original global accuracy: predicted mapping defines the test targets. Unmapped…, datasets: iterable of (dataset name, group key, labeled test loader). Both…, Map each pure predicted group to its truth class; ambiguous merges abstain.…, semantic_alignment(), itertools (+2 more)

### Community 6 - "FL Client"
Cohesion: 0.18
Nodes (6): Client, local_basis(), PACFL structural statistics and clustering, extracted from the baseline., collections, numpy, utils_pacfl_utils

### Community 7 - "Smoke Testing"
Cohesion: 0.16
Nodes (6): Small real training workload for CPU integration checks, not research data., TinyClassifier, TinyDiscriminator, TinyGenerator, torch_utils_data, Train global classifier from group generators and a chosen relation table.

### Community 8 - "Legacy Execution Tests"
Cohesion: 0.15
Nodes (4): difference(), execute_pair(), record(), states()

### Community 9 - "Plans and Methods"
Cohesion: 0.38
Nodes (7): GeFL, image_bi, PACFL, image_bi, ImageBiMapping, GeFL_gan_pacfl_iid, image_bi

## Knowledge Gaps
- **9 isolated node(s):** `by_class`, `ImageBiMapping`, `ByClassMapping`, `GeFLAggregation`, `pure_semantic_class_only` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 94 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `execute_pair()` connect `Legacy Execution Tests` to `Core Networking`, `Model Aggregation`, `CLI Config`, `PACFL Clustering`, `Main & Mapping Inputs`, `Evaluation Metrics`?**
  _High betweenness centrality (0.169) - this node is a cross-community bridge._
- **Why does `IntegrationTests` connect `CLI Config` to `Core Networking`, `Model Aggregation`, `PACFL Clustering`, `Main & Mapping Inputs`, `Evaluation Metrics`, `FL Client`, `Integration Tests`?**
  _High betweenness centrality (0.151) - this node is a cross-community bridge._
- **Why does `Client` connect `FL Client` to `Core Networking`, `Model Aggregation`, `CLI Config`, `Main & Mapping Inputs`, `Smoke Testing`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `execute_pair()` (e.g. with `GeFLAggregation` and `PACFL`) actually correct?**
  _`execute_pair()` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `IntegrationTests` (e.g. with `GeFLAggregation` and `Client`) actually correct?**
  _`IntegrationTests` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `Client` (e.g. with `ClientUpdate` and `GANState`) actually correct?**
  _`Client` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `Server` (e.g. with `AggregationStrategy` and `ClientUpdate`) actually correct?**
  _`Server` has 12 INFERRED edges - model-reasoned connections that need verification._