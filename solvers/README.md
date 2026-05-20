# solvers/

Đặt toàn bộ thuật toán và helper code của nhóm vào thư mục này.

Gợi ý tổ chức:

- `greedy_bfs.py`
- `vrp_ortools.py`
- `aco.py`
- `mapd_cbs.py`

Ưu tiên giữ mỗi solver là một module rõ ràng để dễ test, review, và so sánh kết quả.


## Nhịp làm việc khuyến nghị

Sau mỗi thay đổi nhỏ trong một solver, chỉ chạy smoke test local vài giây:

```bash
bash scripts/run_local_test.sh GreedyBFS
```

Khi đã qua smoke test và muốn so điểm thật, mới đẩy lên Kaggle để chạy `test_config.txt`/full map.
