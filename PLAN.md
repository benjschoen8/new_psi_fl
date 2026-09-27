# image_bi 重構計畫

目標：以 `trainer/GeFL_gan_pacfl_iid` 為算法基準，將控制流程移至新 main，原檔不動。

已定需求：PACFL → GeFL；各 client 本地訓練後方可聚合；訓練前初始化聚合不在此次範圍；協定名 `image_bi`；映射策略含 `image_bi`、`by_class`。

## 設計

- `main.py`：可注入 clustering、aggregation、mapping、global trainer、evaluator；負責呼叫、順序與紀錄。
- `client.py`：獨立 client，本地 GAN/classifier 訓練、PACFL basis 與模型訊息；不知 server。
- `server.py`：僅接受更新資料及聚合策略，保存各群 G/D；不持有 client、測試 loader 或評測方法。
- `clustering.py`：沿用 PACFL adjacency / average linkage，basis 在 client 端產生。
- `mapping.py`：沿用雙向 entropy/cycle mapping；每次運算獨立 image cache；by_class 從 dataset 語義 metadata 建立真值。
- `training.py`：合成資料訓練 global classifier，與 server 分離。
- `evaluation.py`：relation pair metrics，以及以真值語義評估 global classifier；錯誤合併不能因沿用預測 mapping 作標籤而獲得正確分數。
- `setup.py`：建立資料、模型、CLI 設定與 checkpoint 輸出。
- `contracts.py`：明確的模型更新、群模型與 mapping 輸入資料型別。

## 執行與驗證

1. 先寫測試：client/server 解耦、加權聚合、PACFL 分群、label permutation、錯誤合併與未知類別評測、策略替換、訓練後才聚合。
2. 執行 `python -m unittest discover -s revised_protocol/tests -v`，確認新模組尚不存在而失敗。
3. 建立上述模組；重用原神經網路與數值算法，不繼承原 server；新 client 不繼承混合職責的 Node。
4. 加入無須下載資料的 CPU smoke test，實際訓練 GAN、global classifier，覆蓋兩種 mapping。
5. CLI 支援原 YAML 資料設定，輸出設定、群分配、relation tables、分階段時間、模型訊息 bytes、mapping metrics、global accuracy 及 checkpoint。
6. 完成測試與 README；記錄未執行全量 GPU 實驗的限制。

模型訊息 bytes 為 tensor payload 大小估計，並非網路封包量。時間在 CUDA 上同步後量測。初輪所有 client 本地訓練以建立各群模型，後續依 sample_frac 抽樣。

聚合同群 GAN 須有相容 tensor shapes 與相同本地類別順序；若 PACFL 混入不相容 label space，報錯，不以真值偷偷修正分群。真值類別名稱保留大小寫（EMNIST `A` / `a` 不可混同），並按 dataset 原始 label 順序及 label permutation 對應，不排序名稱改動 index。

此資料夾非 Git repository；無 commit/worktree 操作。

## 完成紀錄

上述模組、CLI、兩種 mapping、依賴注入與文件已建立。17 項自動測試、兩種 mapping CLI、pretrained CLI 已驗證；完整證據與限制見 `VALIDATION.md`。
