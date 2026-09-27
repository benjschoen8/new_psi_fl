# 驗證紀錄

日期：2026-09-25。Python 3.12；PyTorch 2.14.0；CPU。

補充：新舊三輪端到端 trace 已發現抽樣、RNG 與預設設定等差異，詳見 [TRACE_REPORT.md](TRACE_REPORT.md)。下列測試通過不代表與舊版逐輪執行等價。

## 自動測試

```bash
MPLCONFIGDIR=revised_protocol/.venv/mpl-cache \
  revised_protocol/.venv/bin/python -m unittest discover -s revised_protocol/tests -v
```

22 項通過。涵蓋 weighted aggregation、PACFL、server 解耦、訓練先於聚合、未抽樣群保留、無 alias 的模型快照、label permutation、EMNIST 大小寫、跨群雙向 cycle 成功與失敗、兩種 entropy filter、錯誤合併的真值評測、mapping 隨機性隔離、設定參數及未訓練模型檢查。包含兩項直接對照原版 mapping metrics 公式與 MCC 零分母規則；已補回新版原先遺漏的 specificity、balanced accuracy 與 MCC。另有三項 old_acc 測試，對照原 server 的整體與各 dataset CSV、漏標籤/漏群排除規則，以及空 mapping 的零分母處理；錯誤合併案例同時斷言 old_acc=1、ground_truth_acc=0。

其中 DCGAN 對照測試使用原 `GeFL_gan_pacfl_iid.Client` 的兩個本地訓練方法，與新 client 在相同模型、輸入、optimizer、seed 下比較 classifier/G/D 全部 state_dict tensors，結果相同。這是本地訓練驗證，不代表整個新舊協定逐 bit 相同。

## CLI 流程

以下三個目錄內均有兩輪 metrics、config、relations、checkpoint 與 log：

- `runs/verification_image_bi/`：新初始化 client、PACFL、GeFL、image_bi、global training/evaluation。
- `runs/verification_by_class/`：同設定的 by_class oracle 流程。
- `runs/verification_pretrained/`：載入第一項 client weights；第一輪不再本地訓練，直接聚合。

新增 `runs/verification_old_acc/`：兩輪皆於 JSON 與 log 並列 old_acc、ground_truth_acc。歷史驗證目錄不回填新欄位。

每輪均評測 16 個合成樣本。另直接比較前兩個 CLI checkpoint：所有 client classifier/G/D tensor 均完全相同，確認策略選擇沒有擾動 client 訓練。

CLI smoke 的 PACFL 結果為單群；跨群 image_bi 的推論與 cycle correctness 由獨立整合測試覆蓋。合成資料只驗證流程，不提供研究成效證據。

## 未驗證

- 真實資料完整訓練、CUDA 行為與論文多 seed 統計。
- 舊版 checkpoint 轉換、完整訓練中斷續跑。
- 任意 peer-to-peer mapping、跨異質 label space 聚合。

評測規則、資料分割範圍、計時與 bytes 估計限制詳見 README。
