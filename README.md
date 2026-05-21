# MAPD Graph Shopee

Repo nền cho bài MAPD Graph Shopee. Mục tiêu của repo này là:

- code thuật toán trong `solvers/`
- làm việc chung qua Git + VS Code
- giữ phần chạy đánh giá Kaggle tách biệt, không sửa file grader

## Cấu trúc chính

- `solvers/`: nơi đặt các solver và helper code
- `run_test.py`: grader/test harness lấy từ Kaggle, không sửa
- `test_config.txt`: config test lấy từ Kaggle, không sửa
- `smoke_config.txt`: config mini cho local smoke test, không dùng để so chất lượng thuật toán
- `smoke_suite_config.txt`: bộ smoke đa dạng hơn gồm case cố định và case ẩn tham số sinh đơn
- `scripts/run_local_test.sh`: entrypoint local mặc định, chỉ chạy smoke test
- `scripts/run_smoke_suite.sh`: chạy bộ smoke đa dạng hơn, vẫn giữ runtime vài giây
- `results/`: output sinh ra khi chạy test local, không commit
- `submission/`: bản đóng gói để nộp Kaggle nếu cần

## Quy ước làm việc

- Mỗi nhánh Git nên bám một việc nhỏ: một solver, một tối ưu, hoặc một bugfix
- Không sửa trực tiếp file grader/config gốc trừ khi đúng luồng nộp bài
- Notebook Kaggle chỉ nên là lớp gọi lệnh, logic chính để trong `solvers/`
- Luồng chạy chính thức phải **offline-ready**: không phụ thuộc `pip install`, `npm install`, tải file, hay gọi internet lúc chấm

## Chạy thử local

Máy local chỉ dùng để bắt lỗi nhanh. Mặc định chạy smoke test vài giây:

```bash
bash scripts/run_local_test.sh
```

Hoặc chỉ định solver đang sửa:

```bash
bash scripts/run_local_test.sh GreedyBFS
bash scripts/run_local_test.sh VRPOrToolsSolver
bash scripts/run_local_test.sh all
```

Smoke test dùng `smoke_config.txt`, chỉ để bắt lỗi import/action/vòng lặp, **không dùng để đánh giá chất lượng thuật toán**.

Khi cần kiểm tra nhiều dạng map nhẹ hơn trước khi đưa lên Kaggle:

```bash
bash scripts/run_smoke_suite.sh GreedyBFS
```

Bộ suite này có `FIXED_*` để bắt lỗi cố định và `RAND_*` để env tự resolve tham số sinh đơn ẩn theo seed/config name. Solver vẫn không được đọc các tham số đó.

Các lượt chạy nặng trên `test_config.txt` hoặc full map nên để Kaggle xử lý:

```bash
python run_test.py --method GreedyBFS --config test_config.txt --out results/
```

## Gợi ý VS Code

- mở root repo này bằng VS Code
- cài extension Python và Jupyter
- bật lưu file tự động nếu team thích workflow nhanh
