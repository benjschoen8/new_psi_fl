# image_bi：PACFL → GeFL

以 `trainer/GeFL_gan_pacfl_iid/` 為算法基準，將協定編排、client、server、relation table 與評測拆開。舊版程式不變。此處的協定名與 CLI mapping 名均採 `image_bi`。

## 職責與流程

```text
main.run
  ├─ client.basis → PACFL → client 分群
  ├─ 第 1 輪：各 client 訓練 classifier、G、D → 匯出更新 → server 聚合
  ├─ 後續輪：main 分送群 GAN → 抽樣 client 本地訓練 → server 聚合
  └─ 到 mapping_round：
       ├─ 獨立 mapping 策略 → relation table（建立一次）
       ├─ 獨立 global trainer → 合成影像訓練 global classifier
       └─ 獨立 evaluator → mapping 指標與 global classifier accuracy
          （其後每輪更新 global classifier 並評測）
```

server 不持有 client、資料 loader 或 accuracy 程式；只儲存與聚合 `ClientUpdate`。client 不引用 server。兩者之間的模型分送、訓練呼叫與傳回更新均由 main 完成。

第一輪所有 client 先完成本地訓練，再聚合。後續才按 `sample_frac` 抽樣。尚未加入「訓練前的 round 0 初始化聚合」；未抽中的群保留前次 GAN。classifier 與 generator/discriminator 架構可不同，但同群內要聚合的 GAN 必須相容。

| 檔案 | 職責 |
|---|---|
| `main.py` | 可注入依賴的協定流程，逐階段計時與紀錄 |
| `client.py` | PACFL basis、本地 GAN / classifier 訓練、收送模型 |
| `server.py` | 保存群 GAN 與呼叫聚合策略 |
| `aggregation.py` | 依樣本數加權聚合 G / D |
| `clustering.py` | label 頻率分配 SVD basis、沿用 PACFL adjacency 與 average linkage |
| `mapping.py` | `ImageBiMapping`、`ByClassMapping` 與 relation table 驗證 |
| `training.py` | 根據 mapping，以群 generator 的合成影像訓練 global classifier |
| `evaluation.py` | relation pair metrics、依真值對齊後的 classifier accuracy |
| `contracts.py` | 型別與獨立的 tensor state 訊息 |
| `setup.py` | CLI、既有 dataset / model factory 整合、模型與紀錄輸出 |
| `smoke.py` | 小模型、合成資料的 CPU 流程測試 |

## 執行

從**本資料夾根目錄**執行，Python 3.10 以上。執行所需模組均在根目錄；`legacy/` 僅供參照，只有與舊版比對的測試會讀取。

```bash
python -m pip install -r requirements.txt
python -m main --help

# 無須下載 dataset 的流程測試
python -m main --smoke --mapping image_bi
python -m main --smoke --mapping by_class

# 實際資料實驗；dataset 必須已備妥
python -m main --device cuda:0 --seed 15698 \
  --mapping image_bi --exp-conf config.yaml

# 相同設定下的 oracle baseline
python -m main --device cuda:0 --seed 15698 \
  --mapping by_class --exp-conf config.yaml

python -m unittest discover -s tests -t . -v
```

預設使用 MNIST、EMNIST、CIFAR10，各 10 個 client。可用 `--num-train-mnist` 等參數變更；亦接受原有 `--num_train_mnist` 寫法。`--rounds`、`--mapping-round` 覆寫 YAML 對應值。`--output` 須為尚不存在的目錄，避免覆寫實驗。

新 YAML 保留 `configs/het-iid-exp.yaml` 的 `dirichlet_alpha: 0.1`。原設定雖名為 iid，實際仍是此 Dirichlet partition；不要僅靠檔名判斷分布。若需較接近 IID 的分布，可另設 alpha，並將其列為實驗條件。

## 真值與 accuracy 定義

`by_class` 以每個 client **實際 local label index** 對應的語義類別建立 oracle relation table。支援 dataset 包裝的 label permutation，包含現有 USPS shuffle。保留類別大小寫，不把 EMNIST `A` 與 `a` 合併，也不排序 class names 而錯置 index。

真值目前以相同的精確類別名稱代表相同語義。若不同 dataset 用別名描述同類（例如 `automobile` / `car`），應在傳入 `label_spaces` 前明確制定 canonical names；程式不自行猜測。真值不依 classifier 的推論結果建立。

`ImageBiMapping` 沿用原 `label_mapping()` 的 entropy filtering 與雙向 cycle 檢查。傳入的類別名稱是匿名 index tokens，策略不取得真實語義名稱。影像 cache 屬於單次 mapping，避免跨實驗殘留。生成 noise 維度來自 config，不再硬編碼為 128。

評測分開保存：

- **Mapping 指標**：不同群間所有 local-label pairs 的 TP / FP / TN / FN、precision、recall、specificity、F1、MCC、balanced accuracy、pair accuracy 與 pair 數。`balanced_accuracy=(recall+specificity)/2` 對應原版 `AvgAccuracy`；`mcc` 沿用原版 MCC 公式與零分母回傳 0 的規則。`pair_accuracy=(TP+TN)/pairs` 是另一個普通 accuracy，不可取代原版 AvgAccuracy。相同 truth、pair 範圍及預測下，已用原函式對照驗證。只有一群時無跨群 pairs，各指標為 0，並有 `pairs=0`，不代表 mapping 正確率為 0。
- **Global classifier accuracy**：先依 relation table 的成員與真值，將模型輸出 global ID 對齊至語義。若某輸出類別錯誤合併了多個真實類別，其預測全部記錯；ID 只是重新編號不會扣分。同一真實類別若被拆為多個純類別，classifier 語義 accuracy 可以仍高，但 mapping recall 會下降，故兩類指標應一起報告。

此 accuracy 明記 `alignment_policy=pure_semantic_class_only`，是嚴格的語義評測定義，**不是舊版 accuracy 的逐值重現**，也不是以測試集最佳指派得到的分數。舊版拿預測 mapping 本身轉換測試標籤，可能使錯誤類別合併仍獲高分。若論文需要其他對齊準則，可注入另一 evaluator 並明示定義。

### 保留原論文的舊 accuracy

每輪 `metrics.jsonl` 的 `evaluation` 及其 `by_dataset` 會同時輸出：

- `old_acc`：沿用舊 server 的定義，以**預測 mapping** 轉換測試標籤後，比較 classifier output ID。不存在於預測 mapping 的 labels/groups 不計入舊分母；`old_correct`、`old_samples` 可供稽核。沒有可評測標籤時回傳 0。
- `ground_truth_acc`：前述真值語義評測；`correct`、`samples` 對應此指標。原有 `accuracy` 欄位仍是此值的相容別名。

兩者使用**同一次 forward pass 的相同 predictions**，不因各跑一次模型或資料增強造成差異。皆儲存為 0–1 fraction；乘 100 後才與原版百分比 CSV 比較。舊 CSV 會四捨五入至小數點後兩位，新 JSON 保留完整精度。`run.log` 亦並列兩個具名指標。

`old_acc` 只保留舊評分規則，不代表新訓練流程已逐值重現舊論文；抽樣與 RNG 差異見 `TRACE_REPORT.md`。既有歷史 run 檔不會自動補上新欄位，須重新評測。同樣本錯誤合併時，`old_acc=1`、`ground_truth_acc=0` 是預期差異。

預設測試範圍沿用基準版本：參與訓練 clients 的 test partitions，micro-average 按樣本數加權，另存各 dataset accuracy；不包含 `num_new_clients` 保留的新 clients，也不是重新跑完整官方 test split。真值 accuracy 將所有供給的測試樣本計入分母；`old_acc` 沿用舊版漏 mapping 時排除該樣本的規則。未知真值 label 會報錯。各 test partitions 應互斥；自訂 loader 若重複資料，會重複計數。

## 替換策略（輕量 DI）

使用一般物件或 callable 注入即可，無 DI framework：

```python
from main import run, RunConfig
from server import Server
from aggregation import GeFLAggregation
from clustering import PACFL
from mapping import ImageBiMapping

# clients、label_spaces、test_sets、兩個 factory 與 trainer 由實驗 setup 建立。
result = run(
    clients,
    Server(GeFLAggregation()),
    clustering=PACFL(threshold=20),
    mapping_strategy=ImageBiMapping(logger, device="cuda:0"),
    global_trainer=trainer,
    generator_factory=generator_factory,
    label_spaces=label_spaces,
    test_sets=test_sets,
    config=RunConfig(device="cuda:0"),
)
```

- clustering：`bases[client_id] → groups[client_id]`。
- aggregation：`list[ClientUpdate] → dict[group, GANState]`，由 `Server(strategy)` 注入。
- mapping：`MappingInputs → RelationTable`；包含群 generators、classifier snapshots、local 類別數，不含 server/client 控制物件。`'by_class'` 是內建 oracle 選項。
- evaluator：`(model, test_sets, predicted_mapping, ground_truth, device) → metrics`。

原本依 server 執行的 mapping 現在取得 server **匯出的模型快照**，其資料需求不變，而 common interface 不被 server 綁定。未來必須直接使用 server 的 adapter 可在建構時注入 server；client 間 mapping 可在策略中注入其通訊介面，仍回傳相同 relation table。此次尚未實作分散式 transport 或新的 peer-to-peer 算法。

PACFL 群內若出現不同 label 順序或語義，程式會拒絕聚合；不以真值替你改寫分群。這是原版「以群首 client 的 label space 代表全群」假設的明確檢查。若要研究可跨異質 label space 聚合的新方案，需一併定義 generator 條件 label 的對齊，不能只移除檢查。

## 論文比較與輸出

每次輸出至 `runs/<timestamp>_<mapping>/`：

- `config.json`：協定、策略、seed、設定、PyTorch 版本、client label metadata、實際 split indices。
- `metrics.jsonl`：每輪 client 選擇、本地 loss、各階段秒數、mapping 指標、global accuracy、模型 payload bytes。
- `relations.json`：群分配、預測 mapping、ground truth。
- `checkpoint.pt`：各 client classifier/G/D、群 GAN、global classifier weights 與語義 metadata。
- `run.log`：執行紀錄。

local training、mapping、global training 分別設定由 seed 衍生的隨機種子，避免某 mapping 多抽了亂數而改變下一輪 client 訓練。CUDA 計時在階段前後同步；秒數不含所有資料搬移、匯出與存檔成本，不能當完整端到端耗時。bytes 是 tensor payload 估算，含 G/D、PACFL basis 及首次 mapping 的 classifier snapshots，**不是實測網路流量**，不包含封包、序列化與自訂 peer-to-peer 通訊。

比較前需保持相同 seed、split、client 模型初始化、本地訓練預算、mapping 時點與 global trainer 預算；報告多 seed 結果。不同 relation table 類別數或編號會改變 global classifier 的輸出空間，即使相同 seed，也不保證 global accuracy 相同。

已有訓練後的 client 模型，可使用此版本的 checkpoint：

```bash
python -m main --mapping by_class \
  --pretrained runs/EXPERIMENT/checkpoint.pt
```

須提供與儲存時相同的 dataset、split、seed、模型參數及 smoke/非 smoke 模式；CLI 驗證 metadata 後載入 classifier/G/D。第一輪直接聚合已訓練的 client weights，其後正常訓練。這是 warm start，**不恢復 optimizer、輪數或 RNG 狀態**；舊版 checkpoint 需另行轉換，未宣稱相容。

## 驗證界線

### Setup-only smoke benchmark (no training)

```bash
python -m setup_smoke --out setup_results
python -m setup_smoke --data-root data/raw --repeats 3 --out setup_results_real
# Small synthetic check, with no dataset or MPC download:
python -m setup_smoke --data synthetic --simulate --clients 3 --labels 3 --out setup_results_synthetic
# Optional: select an existing MP-SPDZ installation and include fuzzy matching
MPSPDZ=/path/to/mp-spdz python -m setup_smoke --methods plain exact fuzzy --out setup_results_mpc
```

By default this uses real MNIST, EMNIST **byclass**, and CIFAR-10, directly reusing
`fl_datasets.load_partitioned_datasets`, its image transforms and partitions, and the original
`setup.label_names` / `setup.label_samples` components. The training entry point uses the same
sampler. Existing datasets and split caches are reused; missing datasets are downloaded by
the original loader. No generator/classifier initialization, warm-up, training, or evaluation runs.

The default compares plain and secure exact circuit setup for 3, 5, 10, 30, and 50 **total**
clients. MNIST/EMNIST/CIFAR-10 receive respectively 1/1/1, 2/2/1, 4/3/3, 10/10/10, and 17/17/16
clients. Real label spaces come from the partitions; `labels_per_client` reports the maximum
(MPC padding size), with minimum/mean also recorded. Up to 16 images per label are sampled.
The original fallback for declared labels without samples is retained. Use `--exp-conf`,
`--seed`, `--noniid-partition`, `--class-subsets`, and `--class-share` to match a training run.
Real data defaults to 20 bucket bits; optional `--data synthetic` defaults to 16 bits and
supports `--labels` (3 by default). Compression does not affect setup, so it is not duplicated.
Fuzzy matching reuses `rt_descriptions.keyword` and `--fuzzy-langs en0,en1`, plus the existing
encoder cache or sentence-transformers.

`setup.csv` contains one row per trial with setup/image/union wall times, union size, estimated
upload/download bytes per client and in total, and any measured MP-SPDZ compilation time,
execution time, and global MB. Data loading/partitioning and image sampling times are separate
columns, excluded from setup timings; shared data preparation is done once per client-count
configuration and reused by all methods/repeats. `setup.json` also records input metadata, configuration, metric limitations,
and underlying protocol statistics. Each completed trial is saved immediately, with atomic
replacement of each file, so a later failed trial leaves earlier results available.
Reusing an output directory overwrites these two files. TLS certificates are prepared for
the largest client count before timing starts.
Secure runs check `MPSPDZ`, then reuse/download/build MP-SPDZ if needed
(x86-64 Linux uses binaries; ARM64 Linux/GX10 builds from source into `~/.cache/mp-spdz`).
On Ubuntu, missing build dependencies are installed using apt-get; sudo may request your password.
The initial source build can take 10–30+ minutes, and later runs reuse it.
`MPSPDZ_JOBS` controls build parallelism (default 4); `MPSPDZ_BUILD_DIR` overrides the cache directory.
Installation time is excluded from
benchmark timings. Installation failures stop the benchmark; there is no silent simulation
fallback. To explicitly run ideal grouping without installing MPC, pass `--simulate`;
its wall time is **not real MPC time**. Plain-only runs do not require MP-SPDZ.
Byte totals retain the protocol's existing estimates; measured MP-SPDZ traffic is reported
separately. Plain label transport is uninstrumented, hence its reported zero bytes is not a
complete network cost. There is no isolated network-latency measurement. Synthetic smoke
results check the setup pipeline, rather than replace full research benchmarks.

測試包含：CPU 真實梯度更新、原版 DCGAN 本地訓練權重對照、PACFL、雙向 mapping 的成功與失敗 cycle、oracle permutation、錯誤合併評測、未抽樣群保留、tensor snapshot 不共用儲存、mapping 隨機性隔離與 server 職責檢查。

合成 smoke 僅驗證管線與輸出；不代表研究精準度。尚未執行完整真實資料集 / GPU 實驗。
