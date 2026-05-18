# MAPD Graph Shopee

Repo nền cho bài MAPD Graph Shopee. Mục tiêu của repo này là:

- code thuật toán trong `solvers/`
- làm việc chung qua Git + VS Code
- giữ phần chạy đánh giá Kaggle tách biệt, không sửa file grader

## Cấu trúc dự kiến

- `solvers/`: nơi đặt các solver và helper code
- `run_test.py`: grader/test harness lấy từ Kaggle, không sửa
- `test_config.txt`: config test lấy từ Kaggle, không sửa
- `results/`: output sinh ra khi chạy test local, không commit
- `submission/`: bản đóng gói để nộp Kaggle nếu cần

## Quy ước làm việc

- Mỗi nhánh Git nên bám một việc nhỏ: một solver, một tối ưu, hoặc một bugfix
- Không sửa trực tiếp file grader/config gốc trừ khi đúng luồng nộp bài
- Notebook Kaggle chỉ nên là lớp gọi lệnh, logic chính để trong `solvers/`

## Chạy thử local

Khi đã có bộ file Kaggle đầy đủ trong repo, dùng:

```bash
python run_test.py --config test_config.txt --out results/ --seed 42
```

Hoặc chạy script wrapper:

```bash
bash scripts/run_local_test.sh
```

## Gợi ý VS Code

- mở root repo này bằng VS Code
- cài extension Python và Jupyter
- bật lưu file tự động nếu team thích workflow nhanh

