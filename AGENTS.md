# MAPD Graph Shopee — ghi chú nền cho agent sau

> Snapshot gốc đã kiểm tra ngày **2026-05-19**; re-verify v3 ngày **2026-05-20**; update mới nhất v6 ngày **2026-05-21** từ:
> 1. trang Kaggle `minhhhtrann/graph-shopee`,
> 2. bundle public dataset **version 6**,
> 3. ảnh thông báo cập nhật đề do nhóm cung cấp.
>
> Mục tiêu của file này: cho agent mới đủ ngữ cảnh để bắt tay vào code mà không phải đọc lại toàn bộ đề, nhưng vẫn biết đâu là “nguồn sự thật” cần kiểm tra lại trước khi nộp.

## 1. Repo này đang đóng vai gì

- Đây là repo làm việc chung qua Git + VS Code cho bài tập **Tối ưu hóa giao hàng đa tác tử thời gian thực**.
- Logic nhóm tự viết nên nằm trong `solvers/`.
- Kaggle dùng để chạy các thử nghiệm/full map và nộp notebook; repo local dùng để phát triển, review, chia nhánh.
- Sau lần setup ngày **2026-05-19** và update v6 ngày **2026-05-21**, repo local đã có:
  - bundle Kaggle v6 chính thức ở root: `env.py`, `run_test.py`, `test_config.txt`
  - solver mẫu/stub chính thức trong `solvers/`
  - `smoke_config.txt` + script smoke test nhẹ để bắt lỗi local
  - `smoke_suite_config.txt` + `scripts/run_smoke_suite.sh` để chạy thêm vài case local nhẹ:
    `FIXED_*` có tham số sinh đơn cố định, `RAND_*` để env tự resolve tham số sinh đơn ẩn
- Sau đính chính chiến lược ngày **2026-05-20**, mỗi solver nên **tự đứng độc lập trong file của nó**; không dùng helper chung giữa Greedy/VRP/ACO/MAPD-CBS.
- Quy ước hạ tầng của nhóm:
  - **local chỉ chạy smoke test vài giây**
  - các lượt chạy nặng trên `test_config.txt`/full map để Kaggle xử lý

## 2. Dataset Kaggle đã xác minh

- Dataset: `Graph_Shopee`
- Chủ dataset: `Minhhh Trann`
- License: MIT
- Version hiện tại đã kiểm tra: **v6**
- Mốc update của v6: **2026-05-21**
- Re-verify ngày **2026-05-21**: Kaggle public là `currentVersionNumber = 6`.
- Mô tả trang Kaggle ghi đề chính thức, Phase 1/2, thang điểm, ràng buộc chạy và công thức thưởng.

### Bundle public v6 tải được thực tế

```text
env.py
run_test.py
test_config.txt
solvers/
├── aco_solver.py
├── greedy_bfs.py
├── mapd_cbs_solver.py
├── solver.py
└── vrp_ortools.py
```

### File cấm đã kiểm tra

Ngày **2026-05-21**, đã tải lại bundle public Kaggle v6 và đối chiếu byte-for-byte:

| File | Trạng thái | SHA-256 |
|---|---|---|
| `env.py` | Match Kaggle v6 | `6fc2168187477fb4c81d5a814dd917b1f1b6490fb2b5a315c9cf6b8ec14a2051` |
| `run_test.py` | Match Kaggle v6 | `f9509950d319583a236b07283e11964490d7619ccd6e11e32b62ac3c85467564` |
| `test_config.txt` | Match Kaggle v6 | `bd8fea414a730c17c590838bbe1f2ded8f399bee28a7b45ac67117e8e8d3db9f` |

Không sửa 3 file này thủ công; chỉ cập nhật nguyên bản khi Kaggle/giảng viên phát hành version mới hơn.

### Lệch cần nhớ

- Mô tả trang Kaggle nói file được cấp gồm `run_test.py`, `test_config.txt`, `demo_notebook.ipynb`.
- Nhưng zip dataset public v6 tải được ở snapshot này **không chứa** `demo_notebook.ipynb`; thay vào đó có thêm `env.py` và các solver mẫu/stub.
- Mô tả trang Kaggle vẫn ghi `cargo_op = 2 [id]`, nhưng code v6 + ảnh thông báo mới cho thấy semantics thực tế đã đổi thành `cargo_op = 2` để giao nhiều đơn cùng đích; khi có lệch, bám theo **bundle v6 + notebook mới nhất + thông báo giảng viên**, không bám mù vào text cũ.
- Notebook công khai trong Code tab đã kiểm tra:
  - `Graph Shopee v1`
  - ref `minhhhtrann/graph-shopee-v1`
  - notebook version đã kiểm tra: **v2**
  - lần chạy gần nhất đã kiểm tra: **2026-05-15**
  - dùng dataset source `minhhhtrann/graph-shopee`

## 3. Bản chất bài toán

- Grid `N x N`; `0` là ô trống, `1` là vật cản.
- Có `C` shipper, chạy trong `T` bước thời gian, tối đa hóa **net reward**.
- Mỗi đơn hàng:

```text
g_i = <sx, sy, ex, ey, et, w, p>
```

  - `(sx, sy)`: điểm lấy
  - `(ex, ey)`: điểm giao
  - `et`: deadline
  - `w`: khối lượng
  - `p ∈ {1,2,3}`: ưu tiên `1=tiêu chuẩn`, `2=nhanh`, `3=hỏa tốc`

- Thang thời gian đề dùng:
  - `1 giờ = 10` time units
  - `1 ngày = 240` time units

## 4. Semantics môi trường phải bám theo v6

Đây là phần quan trọng nhất. Nếu solver lệch semantics này thì điểm thử local có thể đẹp nhưng sai môi trường chấm.

### 4.1. Online + stateful

- **Chỉ `G` được biết từ đầu.**
- Đơn hàng không sinh sẵn toàn bộ lúc `t=0`.
- `DeliveryEnv` chỉ reveal đơn đang xuất hiện theo từng thời điểm; observation hiện tại chỉ chứa các đơn đã xuất hiện và chưa giao.
- Trong v6, vòng bước của env là:

```text
move -> pickup -> delivery -> t += 1 -> reveal đơn mới cho bước kế
```

- `reset()` reveal batch đầu tại `t=0`.
- Phase 1 không công bố surge/hotspot trong config; nếu thiếu, env tự sinh tham số ẩn ổn định theo seed/config.

### 4.2. Hành động ở mỗi timestep

Mỗi shipper trả về một cặp:

```text
(move, cargo_op)
```

- `move ∈ {S, L, R, U, D}`
- `cargo_op = 0`: không làm gì
- `cargo_op = 1`: pickup
- `cargo_op = 2`: delivery

Ràng buộc update mới:

1. Với **pickup**, chỉ được nhặt **duy nhất 1** gói tại ô đó.
2. Khi nhiều gói cùng ô pickup, env nhặt gói tốt nhất theo:
   - ưu tiên cao hơn,
   - deadline sớm hơn,
   - id nhỏ hơn.
3. Với **delivery**, có thể giao **nhiều đơn** trong cùng timestep nếu chúng có cùng tọa độ đích hiện tại.
4. Trong cùng timestep, mỗi shipper chỉ làm **một trong ba** cargo action: `0`, pickup, hoặc delivery. Không pickup và delivery cùng lúc.
5. Trong code v6, `cargo_op = 2` **không truyền id**; env duyệt toàn bộ `bag` và giao mọi đơn hợp lệ tại vị trí hiện tại.

### 4.3. Va chạm và di chuyển

- Không được đi ra ngoài map hoặc vào vật cản.
- Nếu tranh chấp ô, shipper có id nhỏ hơn được ưu tiên giữ ô.
- Di chuyển chỉ mất phí khi dùng `L/R/U/D`; đứng yên `S` không mất phí.
- Move cost:

```text
-0.01 * (1 + W_carried / W_max)
```

### 4.4. Trọng lượng, sức chứa, thưởng

- Mỗi shipper phải thỏa đồng thời:
  - tổng khối lượng trong túi `<= W_max`
  - số đơn trong túi `<= K_max`
- `r_base(w)`:
  - `w <= 0.2`: `4`
  - `0.2 < w <= 3`: `10`
  - `3 < w <= 10`: `15`
  - `10 < w <= 30`: `20`
  - `w > 30`: `30`
- Nếu đúng hạn:

```text
r = alpha_p * r_base * (1 + bonus)
bonus = max(0, (et - t_delivery) / et)
alpha = {1:1.0, 2:2.0, 3:3.0}
```

- Nếu trễ:

```text
r = beta_p * r_base * max(0, 1 - (t_delivery - et) / T)
beta = {1:0.1, 2:0.3, 3:0.5}
```


### Đính chính surge/hotspot ngày 2026-05-21

- Giảng viên xác nhận: **không được truy cập hoặc dựa vào tham số surge/hotspot trong cả Phase 1 lẫn Phase 2**.
- Câu “Phase 2 công bố surge/hotspot” trong mô tả cũ chỉ được hiểu là config để test/chấm; chương trình solver không được xây tối ưu dựa trên các tham số này.
- Dataset v6 update `env.py` để không expose `cfg/public_cfg`, và `solver.py` bỏ `self.cfg = env.public_cfg`.
- Solver hợp lệ chỉ dùng observation public từ `env.observe()`/`env.step()` và các thuộc tính public cần thiết như `env.grid`, `env.G`, `env.T`, `env.config_name`; không introspect private attrs như `__hotspots`, `__surge_windows`.

## 5. Sinh đơn, surge, hotspot

- Đơn xuất hiện theo Poisson không đồng nhất.
- Ngoài surge: `lambda(t) = lambda0`
- Trong surge: `lambda(t) = lambda0 * (1 + surge_amplitude)`
- Hotspot:
  - trong surge, xác suất `70%` lấy source quanh hotspot Manhattan radius `<= 3`
  - còn lại `30%` rải toàn map
- Phase 1:
  - `surge_windows` và `hotspots` không công bố trong `test_config.txt`
  - env tự resolve tham số ẩn nếu thiếu
- Phase 2:
  - theo đính chính Zalo 2026-05-21, solver vẫn không được truy cập/dựa vào surge/hotspot

### 5.1. Ranh giới thông tin hợp lệ cho solver

- Solver **không được** đọc trực tiếp tham số sinh đơn, surge window, hotspot, lambda, amplitude, hoặc private attrs trong `DeliveryEnv`.
- Không dùng kỹ thuật introspection/name-mangling để truy cập các trường kiểu `__hotspots`, `__surge_windows`, `__lambda0`, v.v.
- Từ v6, `Solver` base class không còn `self.cfg`; mọi thuật toán phải ra quyết định từ observation public hiện tại.
- Có thể dùng các tín hiệu quan sát hợp lệ như:
  - `obs["t"]`, `obs["orders"]`, `obs["new_order_ids"]`, `obs["shippers"]`, `obs["grid"]`;
  - thuộc tính public cần thiết như `env.grid`, `env.G`, `env.T`, `env.config_name`.
- Nếu báo cáo cần nói về surge/hotspot, chỉ mô tả chiến lược **thích nghi từ đơn đã xuất hiện** như nhận diện cụm pickup/delivery từ `orders` hiện tại, không nói là dùng tham số thật của env.

## 6. Các file v6 cần hiểu trước khi sửa solver

### `env.py`

- Là simulator chính, đã chuyển sang online/stateful.
- `DeliveryEnv.observe()` trả:
  - `t, N, C, G, T`
  - `grid`
  - `orders`: chỉ các đơn chưa giao đang lộ diện
  - `new_order_ids`
  - `shippers`
  - `done`
- `Shipper.pickup_best()` đã encode luật “nhặt đúng 1 đơn tốt nhất”.
- `_deliver_many()` encode luật “giao nhiều đơn cùng đích”.

### `run_test.py`

- Chạy mỗi solver trên một env mới cùng seed/config để tránh state leak.
- Có 4 method chính thức:

```text
GreedyBFS
VRPOrToolsSolver
ACOSolver
MAPDCBSSolver
```

- Có cờ:

```bash
--method all
--method GreedyBFS
--method VRPOrToolsSolver
--method ACOSolver
--method MAPDCBSSolver
```

- Ghi output vào `results/`:
  - `result_<config>.json`
  - `summary.json`
  - `all_results.json`
- Giới hạn tổng runtime: `3600s = 60 phút`.

### Notebook Kaggle `Graph Shopee v1`

- Notebook chỉ là lớp orchestration, **không chứa thuật toán**.
- Cell chính của notebook:
  1. copy dataset vào workspace:

```bash
cp -r ../input/datasets/minhhhtrann/graph-shopee ./
```

  2. `cd graph-shopee`
  3. chừa một dòng cho nhóm tự đổi đường dẫn copy folder `solvers`
  4. in SHA-256 của các file trong `./mapd_shopee/solvers`
  5. chạy evaluator:

```bash
python run_test.py --method GreedyBFS --config test_config.txt --out results
```

- Ghi chú của notebook: danh sách file và hash giữa Phase 1 và Phase 2 phải khớp nhau.
- Notebook còn comment sẵn:
  - dùng `--method all` nếu chạy cả 4 phương pháp
  - hoặc truyền đúng một trong `GreedyBFS`, `VRPOrToolsSolver`, `ACOSolver`, `MAPDCBSSolver`
- Có một dấu hiệu lệch đường dẫn cần kiểm tra khi submit:
  - notebook `cd graph-shopee`
  - nhưng cell hash lại trỏ `./mapd_shopee/solvers`
  - nên trước khi nộp thật, hãy mở version notebook mới nhất trên Kaggle và chạy thử đường dẫn copy/hash một lượt, đừng giả định text comment là hoàn hảo.

### `solvers/`

- `greedy_bfs.py`: nhóm đã thay bằng bản Greedy self-contained, chạy online trên observation hiện tại, không đọc cfg/surge/hotspot.
- `vrp_ortools.py`, `aco_solver.py`, `mapd_cbs_solver.py`: mới là stub/TODO trong bundle mẫu.
- `solver.py`: base class v6, không còn `self.cfg`; stubs gọi `default_result(method, env.config_name, env.G, orders)`.

### `test_config.txt` Phase 1 v6

| Config | N | C | G | T | K_max | W_max |
|---|---:|---:|---:|---:|---|---|
| C1 | 7 | 2 | 15 | 240 | `3 3` | `20 20` |
| C2 | 10 | 2 | 25 | 240 | `3 3` | `20 30` |
| C3 | 12 | 3 | 40 | 360 | `3 2 3` | `20 30 20` |
| C4 | 15 | 4 | 60 | 600 | `3 3 2 2` | `20 20 30 30` |
| C5 | 18 | 5 | 80 | 780 | `3 3 2 2 3` | `20 20 30 30 20` |
| C6 | 20 | 5 | 100 | 960 | `3 3 2 2 3` | `20 20 30 30 20` |

- Tất cả map có obstacle/bottleneck.
- Ảnh thông báo của giảng viên nói **`test_config.txt` Phase 1 cũng đã được update** ở lần đổi này; không dùng bản cũ.

## 7. Yêu cầu bài và chấm điểm

### Method phải làm

- Bắt buộc:
  - Greedy BFS — 5 điểm
  - VRP + OR-Tools — 5 điểm
- Nâng cao:
  - ACO — 2.5 điểm
  - MAPD-CBS — 2.5 điểm

### Báo cáo

- Mô tả nguyên lý từng thuật toán.
- Phân tích time/space complexity.
- Ghi mức độ tối ưu: optimal / near-optimal / heuristic.
- So sánh định lượng trên các config Phase 1:
  - net reward
  - % giao đúng hạn
  - runtime
- Có phần phân tích cách ứng phó surge/hotspot là điểm cộng quan trọng.

### Tổng điểm

| Hạng mục | Điểm |
|---|---:|
| Greedy BFS | 5 |
| VRP + OR-Tools | 5 |
| ACO | 2.5 |
| MAPD-CBS | 2.5 |
| Báo cáo kỹ thuật | 5 |
| Ranking Phase 2 | 10 |
| Vấn đáp | 20 |
| Tổng | 50 |

## 8. Phase 1 / Phase 2 và luật nộp

### Phase 1

- Theo phản hồi giảng viên trong chat lớp, deadline Phase 1 là **23h59 ngày 28/05/2026**.
- Nhóm phát triển bằng `test_config.txt`.
- Chạy chuẩn:

```bash
python run_test.py --config test_config.txt --out results/
```

- Upload thư mục `solvers/` của nhóm lên Kaggle dưới dạng Dataset và để public đúng deadline Phase 1.
- Notebook chấm mẫu không được sửa logic; chỉ đổi dòng copy từ dataset private của nhóm vào `solvers/`.
- Kaggle chỉ được nộp **1 version duy nhất**; vi phạm bị trừ điểm.

### Phase 2

- Ba ngày trước deadline, file config ranking sẽ được công bố bằng cách update chính file config cũ.
- Notebook đã nộp không được sửa.
- Ở cuối Phase 1 / khi chạy Phase 2, theo thông báo mới có 2 lựa chọn:
  1. chạy cả 4 method bằng `--method all`
  2. chỉ chạy method tốt nhất bằng tên method cụ thể
- Vì notebook Phase 2 phải chạy độc lập trong 60 phút, chọn method tốt nhất có thể là chiến lược hợp lý nếu chạy cả 4 quá nặng.

### Hạn chế chạy

- Tổng runtime tối đa 60 phút.
- Khi chạy lại không có internet.
- Không được cài thêm thư viện ngoài lúc chạy; đừng phụ thuộc `pip install`.
- Nếu `solvers/` chứa file không phải code, phải mô tả quy trình tạo và nộp kèm bước sinh file đó.
- Tính đến lần kiểm tra local ngày **2026-05-19**, code đang có chỉ dùng Python standard library; `ortools`, `numpy`, `scipy` đều không có trong môi trường local này. Trước khi viết thật nhánh `VRP + OR-Tools`, phải xác nhận `ortools` đã có sẵn trong đúng môi trường Kaggle sẽ chấm, vì final run không được tải/cài package mới.
- Nếu mô tả/bảng trong đề lệch với code được cấp, bám theo `env.py`/mục công thức chính của đề. Giảng viên đã xác nhận bảng chi phí theo hạng cân ở mục 1.4 chỉ là ví dụ; chi phí di chuyển thực tế follow mục 1.6 và code:

```text
move_cost(w_carried, w_max) = -0.01 * (1 + GAMMA * w_carried / max(w_max, 1.0))
```

## 9. Hệ quả kỹ thuật cho code nhóm

1. **Đừng lập kế hoạch trên toàn bộ đơn tương lai.** Solver phải phản ứng online với observation hiện tại.
2. **Luôn coi đơn mới là stream**, không phải list tĩnh từ đầu episode.
3. **Pickup batching không hợp lệ**; chỉ delivery batching cùng destination là hợp lệ.
4. **Destination clustering đáng khai thác** vì `cargo_op=2` có thể xả nhiều đơn cùng đích một lượt.
5. **Deadline + priority phải đi vào scoring**, vì pickup tại một ô không được tự chọn bừa đơn nếu env luôn lấy best.
6. **Bottleneck/collision awareness** quan trọng từ C3 trở lên; shipper id thấp có lợi khi tranh chấp ô.
7. **VRP/OR-Tools nếu dùng** phải biến thành rolling-horizon/dynamic VRP, không phải solve một lần cho toàn bộ `G`.
8. **Phase 2 ranking** tối ưu theo `net_reward`, không chỉ delivery rate.
9. **Giữ notebook mỏng**; mọi logic thật để trong repo/`solvers/`, vì đây cũng là cách team đang phối hợp qua Git.
10. **Không trói các thuật toán bằng helper chung**: sau Phase 1 có thể chỉ chọn một method để chạy Phase 2, nên mỗi solver nên tự chứa logic cần thiết của nó. Greedy BFS hiện tự chứa BFS/scoring/conflict local trong `greedy_bfs.py`.

## 10. Checklist handoff nhanh cho các bạn cùng nhóm

Phần này dành cho người/agent mới vào repo, đặc biệt khi đang vibe coding để không vô tình phá luật chấm.

### Luật không được vi phạm

- **Không sửa thủ công** `env.py`, `run_test.py`, `test_config.txt`. Nếu Kaggle ra version mới thì update nguyên bản và ghi lại hash.
- **Không đọc thông tin ẩn**: không truy cập `cfg/public_cfg`, private attrs của env, surge/hotspot/lambda/amplitude thật.
- **Không dùng internet/cài package lúc chạy**: không `pip install`, `npm install`, `curl`, `wget`, `requests`, v.v. trong luồng chấm.
- **Không fit cứng public test**: tránh logic kiểu `if config_name == "C6"`; nếu cần adaptive thì dùng đặc trưng tổng quát như `N`, `C`, `T`, bag size, visible orders, obstacle pattern public.
- **Không tạo helper chung bắt buộc giữa 4 solver**. Mỗi solver phải tự chạy được nếu Phase 2 chỉ chọn đúng method đó.

### Hiểu đúng điểm chạy

- `net_reward` là điểm objective của simulator, không phải điểm môn.
- `run_test.py` cộng `net_reward` qua các config để ra `total_score_by_method`.
- Public `test_config.txt` hiện có C1-C6 để dev/báo cáo; Phase 2 có thể update config ranking mới. Vì vậy tối ưu public score nhưng vẫn phải giữ thuật tổng quát.
- Smoke test chỉ bắt lỗi import/action/loop, **không** dùng để kết luận thuật mạnh hay yếu.

### Hiểu đúng thời gian và số đơn

- `T` là số timestep trong game/simulation, không phải giây thật.
- `G` là tổng số đơn hữu hạn trong config; đơn được reveal dần, solver chỉ thấy đơn đã lộ diện.
- Giới hạn `3600s` là wall-clock runtime thật của notebook/evaluator. Không cần dùng hết 1 tiếng; càng nhanh càng an toàn.
- Episode thường chạy đến `t >= T`; nếu giao hết sớm thì solver vẫn nên trả action hợp lệ/đứng yên đến hết mô phỏng.

### Chính sách dependency cho VRP + OR-Tools

- Local hiện tại **không import được `ortools`**. Không giả định Kaggle có sẵn nếu chưa check trực tiếp trong notebook chấm.
- Không vendor/copy OR-Tools binary vào `solvers/` trừ khi giảng viên xác nhận rõ là được phép. Đây là vùng xám vì OR-Tools có native binary, nhiều file, phụ thuộc Python/Kaggle image.
- `vrp_ortools.py` nên có fallback stdlib:
  - nếu import được OR-Tools thì dùng rolling-horizon VRP nhỏ;
  - nếu không import được thì dùng VRP-inspired greedy insertion/assignment để method vẫn chạy offline.

### Khi sửa solver

- Greedy BFS đang là solver thật, self-contained, không đọc thông tin ẩn; giữ tính chất này khi chỉnh tiếp.
- Chỉ dùng `test_config.txt` để regression/probe, không ghi con số benchmark vào luật agent và không hardcode theo public config.
- Khi tối ưu, ghi chú lý do ở mức thuật toán tổng quát: scoring, assignment, route ordering, conflict handling, fallback dependency.

## 11. Việc agent sau nên làm đầu tiên

1. Kiểm tra Kaggle có version mới hơn v6 chưa.
2. Nếu Kaggle chưa đổi version, chạy smoke test local trước khi sửa thuật toán:

```bash
bash scripts/run_local_test.sh
```

3. Khi sửa một solver cụ thể, chỉ smoke đúng solver đó local:

```bash
bash scripts/run_local_test.sh GreedyBFS
```

4. Khi cần kiểm tra đa dạng hơn nhưng vẫn nhẹ:

```bash
bash scripts/run_smoke_suite.sh GreedyBFS
```

5. Khi cần điểm thật/baseline thật, chạy trên Kaggle bằng `test_config.txt`; không biến local thành runner full map.
6. Nếu notebook Kaggle mới nhất có tên method/luồng nộp khác, cập nhật note này trước rồi mới tối ưu tiếp.
