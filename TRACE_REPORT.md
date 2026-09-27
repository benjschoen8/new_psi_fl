# 新舊協定執行對照

2026-09-25。比較原版 `trainer/GeFL_gan_pacfl_iid` 與 `revised_protocol` 的 `image_bi`。

後續更新：新版已並列輸出原版定義的 `old_acc` 與真值定義的 `ground_truth_acc`，並用相同 predictions 對照原 server 驗證。下文對抽樣、RNG 與整體不等價的結論仍成立；「新版 accuracy」在原 trace 中指真值指標。

**結論：主要算法與階段順序一致，但目前不是執行結果等價的重構。相同初始模型、資料與 seed，在第 1 輪本地訓練後即出現不同權重。先前 17 個測試通過，只代表各項功能成立，不能據此宣稱新舊端到端等價。**

## 實際 trace 範圍

執行 `tests/trace_legacy.py`：4 clients、2 個 PACFL 群、3 輪、mapping 在第 2 輪；分別測 `sample_frac=1.0` 與 `0.5`。原版 `Server.run()`、`Client.update()`、aggregation、mapping、global training、accuracy 實際執行；僅以小型神經網路取代模型 factory，避免完整 ResNet / GAN 訓練成本。新版本則直接執行 `main.run()`。

另分階段控制 RNG / 輸入，單獨檢查 PACFL、聚合、global training 與 accuracy 定義。

```bash
MPLCONFIGDIR=revised_protocol/.venv/mpl-cache \
  revised_protocol/.venv/bin/python -m revised_protocol.tests.trace_legacy
```

本次原始事件與權重差異：[`trace.json`](runs/legacy_trace_licmku7n/trace.json)。每次重跑另建獨立目錄。

## 確認一致的部分

| 階段 | 結果 |
|---|---|
| 初始 client classifier/G/D | 對照 fixture 全部 tensor 完全相同 |
| PACFL basis | 相同 loader 順序與 RNG 下，basis 完全相同 |
| PACFL clustering | 此 fixture 同為 `{0,1}`、`{2,3}` 兩群 |
| 本地訓練數學 | 既有測試以真實 DCGAN 與原版 methods 比較，所有權重相同 |
| 聚合 | 同為按樣本數加權；本例最大 float32 差 `1.1920928955078125e-07`，在 `1e-6` 容差內 |
| classifier 停訓時點 | mapping_round=2：第 1、2 輪訓練 classifier，第 3 輪只訓練 GAN |
| mapping 呼叫時點 | 第 2 輪聚合後執行一次 |
| image-bi 判定 | 新版直接使用原版 `label_mapping()`，entropy 與 cycle 判定相同；生成影像 RNG 則不同，見下文 |
| global training 時點 | 第 2、3 輪各執行一次，保留 optimizer / model 狀態 |
| global training 數學 | 固定相同 generator outputs、初始化與 RNG 消耗後，global classifier tensors 完全相同 |
| 未抽中群 | 保留上一輪 GAN |
| 控制權 | 新版由 main 呼叫 client/server；server 不持有 client 或 evaluator |

## 阻止等價重現的差異

### 1. Client 抽樣與第一輪訓練預算改變

原版 `trainer/BaseFL/server.py:86` 每輪使用 `np.random.choice`，再排序 ID；第一輪同樣遵守 sample_frac。

新版 `main.py:88` 第一輪強制全員，後續以独立的 Python `random.Random.sample` 抽樣，且不排序。

實測 `sample_frac=0.5`：

| 輪 | 原版 | 新版 |
|---|---|---|
| 1 | `[1,3]` | `[0,1,2,3]` |
| 2 | `[0,3]` | `[3,2]` |
| 3 | `[0,1]` | `[2,1]` |

新版第一輪訓練量是原版的兩倍。即使 sample_frac=1，新版後兩輪順序為 `[3,2,1,0]`、`[1,0,2,3]`，原版保持 `[0,1,2,3]`。全員預訓練可作明確的新協定，但不能算保持原流程不變。

### 2. Seed 管理與隨機數消耗不同

原版只於實驗起點設 seed，此後共用連續 RNG；新版 `main.py:51` 對 basis、clustering、各 client 各輪、mapping、global training 重設衍生 seed。

實測全員參與：初始模型完全相同，第一輪結束最大 client 權重差已為 `0.0011898130178451538`；第三輪為 `0.0017491206526756287`。此差異不是聚合的 `1e-7` 捨入誤差。

另外，原版 `get_gen_images()` 在 GAN cache miss 時先丟棄一次 `randn`，再取第二次 noise；新版只抽一次。原版 global trainer 每輪重建 generators，新版重用當輪已具現化的 generators；這也改變 RNG 消耗。CUDA 下原版部分 noise 先在 CPU 產生再搬到 GPU，新版直接在 device 產生。因此即使只移除 seed_stage，也不足以保證逐 tensor 相同。

真實 PACFL 每類最多取 loader 中前 64 個樣本。新版改 basis seed，可能取到不同樣本而改變分群；本 fixture 每 client 僅 8 筆，不能證明大型資料時的分群完全相同。

### 3. CLI 預設實驗組成不同

原 `main.py:102` 預設六種 dataset 各 10 clients；新 `setup.py:99` 只預設 MNIST、EMNIST、CIFAR10 各 10，FashionMNIST、CIFAR100、USPS 為 0。這會從 60 clients 變成 30，直接改變資料、分群、模型與計算量。

另外原 CLI 預設 algorithm 為 `Ours`，預設 config `configs/het-exp.yaml` 在目前資料夾不存在。要比較此處指定基準，原版必須明傳 `--algorithm GeFL_gan_pacfl_iid`、存在的 YAML、seed、各 dataset client 數及 mapping 時點，不能只比較兩個無參數啟動命令。

### 4. Mapping 時點的設定來源改變

原版 Node 使用 `args.start_mapping_epoch`，CLI 預設 25，YAML 同名值不會覆寫它；新版 `setup.py:124` 未傳 `--mapping-round` 時採 YAML。

例如同傳 `configs/het-noniid-exp.yaml`：其 YAML 值為 1，原版仍預設第 25 輪 mapping，新版變第 1 輪，連帶改變 classifier 停訓與 global training 輪數。比較時必須明確指定一致值。

## 已知、刻意或合理但仍非等價的改動

### 真值與 accuracy

原版 `server.py:519` 用預測 relation table 將測試 labels 轉成 model IDs，再比較相等。新版 `evaluation.py:6` 先依真值語義建立 alignment：錯誤合併多個真實類別的 model ID，其預測全部記錯。

同模型、同樣本、兩個真實類別均被錯誤映射至 output 0 的控制測試：**原版 100%，新版 0%**。這是評測定義的改變，不能作新算法改善或退步的證據。

反之，同一真實語義被拆成多個純 model IDs 時，新版可以接受其中任一 ID；原版要求對應群的那個 ID。故這也不只是「把錯誤合併扣分」的局部修正。新舊報表應分別命名，並獨立報 mapping precision/recall。

此變更符合要求的 ground-truth 評測方向，但具體 `pure_semantic_class_only` 準則是新版選定的研究指標，仍需在論文方法中明述。

### 類別 metadata

原 `utils/train_utils.py:59` 使用排序後 class names，且沒有依 USPS permutation 重排 names。新版依 dataset 真實 local label 順序與 permutation 建立語義，保留 EMNIST 大小寫；這是修正，但不會重現舊 truth 結果。

### 分群不相容時的行為

原 PACFL server 直接以第一個 client 的 label space 代表整群。新版 `main.py:74` 發現同群 ordered label spaces 不同即拋錯。這避免靜默誤配，但真實 PACFL 混群時，可能出現原版繼續、新版停止的差異。

### 支援與輸出範圍

- 新版 CLI 僅提供 image_bi / by_class；原版還有其他 mapping 分支，此次沒有全數移植。
- 舊版 client 可用 `loss` 選 loss class，新版固定 CrossEntropyLoss；此次基準設定本來也使用 CrossEntropyLoss。
- 舊版允許 local_epochs=0；新版要求 classifier/GAN epochs 大於 0。
- `num_new_clients>0` 時，原 client ID 會跨 dataset 留空號，新版連續編號；新 checkpoint 不可直接對應舊 client ID。
- 舊 accuracy CSV 使用百分比且另有 per-class CSV；新 JSON 使用 0–1 fraction，未輸出同格式的 per-class CSV。
- 舊 PACFL `run()` 的 checkpoint 呼叫已註解；新版結束時固定輸出新格式 checkpoint。格式不相容，pretrained 是 warm start，不是 resume。
- 新版保存聚合 state 的整數 buffer 原 dtype；原聚合先保留 float32，load_state_dict 才轉型。一般 BatchNorm 計數器載入效果相近，但原始儲存結果不完全同型別。

## 建議的下一步

若目標是先驗證「只解耦、不改算法」，應先將抽樣、client 順序、CLI 實驗組成、mapping 時點明確對齊；另定是否要求 RNG 序列也完全重現。至少比較每輪 selected IDs、群分配、更新 tensor、relation table、合成樣本與 global classifier logits。舊 accuracy 可保留成獨立的相容指標，新真值 accuracy 另列，不能混用。

本次只新增 trace 程式與此報告，沒有為了通過比較而修改任一版本的協定邏輯。

## 限制

目前沒有 `data/raw` 真實資料，未執行完整 dataset/GPU 訓練。小模型對跑足以證明「不等價」，不足以證明所有配置的算法保真。原 17 項測試仍需搭配本 trace 解讀，而不能代替它。
