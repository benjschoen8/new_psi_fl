"""Label union: build the list of labels that exist across all clients, without revealing who
holds what, and score it.

  discover   one SecAgg of dictionary indicator vectors (--union secagg in secure_code_no_cluster /
             secure_code_cluster): Aggregator learns the union + holder counts, not who
  mpc_union  exact union by all clients in MPC (circuit sort + rank, PSI-style): each client
             learns only slots for its own labels; nobody learns the union (secure_main)
  oprf_union PSI-style union (the pipelines' default, --union oprf): n-party OPRF tags + one SecAgg,
             no dictionary needed; Aggregator gets only the index list, each client its own indices
  dict_union MPC union over the public dictionary (--union mpc): the
             Aggregator gets only the index list 0..U-1, each client the index of its own labels
  metrics    union_metrics: TP/FP/FN/TN over dictionary entries, precision, recall, F1, MCC;
             index_metrics: the same for dict_union's index outputs (+ split/merged indices)
"""
from .discover import indicator, discover
from .metrics import union_metrics, true_union, index_metrics
from .dict_union import mpc_union
from .oprf_union import oprf_union
