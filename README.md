# 📦 Dự báo Doanh số Bán lẻ theo Mặt hàng (Item-level Sales Forecasting)

Dự án dự báo số lượng bán (`TotalQuantity`) theo từng mặt hàng (`ItemCode`) và từng ngày, sử dụng chuỗi thời gian bán lẻ thực tế.  
Metric đánh giá chính: **WRMSSE** (Weighted Root Mean Squared Scaled Error).

---

## 📁 Cấu trúc file

```
├── EDA_classes.py                    # Thư viện EDA dùng chung
├── Multiple_Baseline_Model.ipynb     # Notebook 1: Baseline model (LGBM theo nhóm SKU)
├── Base_Model__Basic_Ensemble_.ipynb # Notebook 2: Single LGBM + Ensemble Model
├── Advanced_Models.ipynb             # Notebook 3: Stacking model nâng cao
└── README.md
```

---

## 📊 Dữ liệu

| File | Mô tả |
|------|-------|
| `train_data.csv` | Dữ liệu giao dịch bán hàng theo ngày |
| `sample_submission.csv` | Template nộp bài thi |

**Các cột chính:** `Date`, `ItemCode`, `Quantity`, `SalesAmount`, `UnitPrice`, `Unit Cost`, `Cost Amount`

**Tập train / validation:**
- Train: đến `2025-08-08`
- Validation: `2025-08-09` → `2025-09-05` (28 ngày)
- Test (Public + Private): 28 ngày tiếp theo

---

## 🧪 EDA_classes.py

Thư viện OOP hỗ trợ các bước EDA, gồm 4 class:

| Class | Chức năng |
|-------|-----------|
| `EDA` | Hiển thị dataframe, info, shape |
| `DuplicateAndNullAnalyzer` | Kiểm tra trùng lặp, giá trị thiếu, null theo nhóm |
| `DistributionAnalyzer` | Mô tả thống kê, boxplot, density, outlier detection |
| `CorrelationAnalyzer` | Ma trận tương quan, phân tích tương quan với target |
| `DistributionComparison` | So sánh phân phối train vs test (KS test, Chi-square) |

---

## 🔧 Tiền xử lý chung (áp dụng cho cả 3 notebook)

1. Loại bỏ duplicate, chuẩn hóa kiểu dữ liệu (decimal dấu phẩy → chấm)
2. Gom nhóm theo `(Date, ItemCode)` → `TotalQuantity`, `TotalSalesAmount`, `TotalCostAmount`
3. Tính `AverageUnitPrice`, `AverageUnitCost`, `DailyProfit`, `ProfitMargin`
4. Clip `TotalQuantity >= 0`

---

## 📓 Notebook 1: Multiple Baseline Model

**File:** `Multiple_Baseline_Model.ipynb`

### Mô hình
- **LightGBMTwinModel** — train 2 model song song (Public / Private branch)
- Phân nhóm SKU (`assign_sku_group`) để train model riêng theo từng nhóm

### Features
- Time: `Weekday`, `Day`, `Month`, `IsWeekend`
- Lag: `lag_28`, `lag_29`, `lag_35`, `lag_56`, `lag_57`, `lag_63`
- Rolling: mean/std với window 7, 14, 30 (lag 28 và 56)
- Profit: `DailyProfit`, `PriceCostRatio`, `ProfitMargin`
- NaN indicator + fill cho lag features

---

## 📓 Notebook 2: Base Model (Basic Ensemble)

**File:** `Base_Model__Basic_Ensemble_.ipynb`

### Mô hình
- **LightGBMTwinModel** đơn (không phân nhóm SKU)
- **Ensemble Model**: Kiến trúc **2 tầng**:

**Tầng 1 — Base Models (2 mô hình):**
| Model | Vai trò |
|-------|---------|
| LightGBM | Bắt pattern phi tuyến, lag/rolling features |
| CatBoost | Xử lý tốt categorical features (ItemCode) |

**Tầng 2 — Meta Model:** `Ridge(positive=True)`


---

## 📓 Notebook 3: Advanced Models (Stacking)

**File:** `Advanced_Models.ipynb`

### Mô hình — `StackingTwinModelV2`

Kiến trúc **2 tầng**:

**Tầng 1 — Base Models (4 mô hình):**
| Model | Vai trò |
|-------|---------|
| LightGBM | Bắt pattern phi tuyến, lag/rolling features |
| CatBoost | Xử lý tốt categorical features (ItemCode) |
| RandomForest | Ổn định, ít overfit |
| Ridge L1 | Bắt xu hướng tăng trưởng tuyến tính |

**Tầng 2 — Meta Model:**
- `Ridge(positive=True)` — kết hợp 4 OOF predictions từ tầng 1

### Features nâng cao (so với baseline)
- `static_return_rate`: tỷ lệ hàng trả lại theo SKU (smoothed)
- `static_sparsity`: độ thưa của chuỗi thời gian
- `static_cv`: hệ số biến thiên (CV) của TotalQuantity
- Cyclic encoding: `sin_doy`, `cos_doy` (ngày trong năm)
- Log-transform lag: `lag_28_log`, `lag_56_log`
- Rolling max: `rolling_max_7_lag28`, `rolling_max_7_lag56`

---


## ▶️ Cách chạy

Tất cả notebook chạy trên **Google Colab**. Dữ liệu được tải tự động từ Google Drive qua `gdown`.

```python
# Cài thư viện cần thiết
!pip install catboost lightgbm

# Chạy từng section theo thứ tự:
# 1. CHUẨN BỊ DỮ LIỆU
# 2. EDA
# 3. Tiền xử lý dữ liệu
# 4. Feature Engineering
# 5. Huấn luyện model
# 6. Infer + tạo submission
```

---

## 🛠️ Thư viện sử dụng

```
numpy, pandas, matplotlib, seaborn
lightgbm, catboost, scikit-learn (RandomForest, Ridge)
scipy, gdown, google-colab
```

---
