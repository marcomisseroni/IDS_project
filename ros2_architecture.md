# ROS 2 Architecture Reference

Working notes on the actual node/topic structure of this repo, built by reading the node
source directly (not from a running graph, not from the README). Meant as context for
writing a report — cross-check anything load-bearing against the files listed inline.

See also: [`README.md`](README.md) (project pitch), [`ros2_commands.md`](ros2_commands.md)
(ROS 2 how-to + real-hardware deployment commands), `IMDCL_paper.pdf` (the localization
algorithm this implements).

---

## 1. What the system actually does

Three LIMO robots (`limo_0`, `limo_1`, `limo_2`) and one tracked person. Each robot detects
ArUco markers on the other robots and on the person via its camera, runs a decentralized EKF
that fuses those relative measurements with its own odometry, and an MPC controller drives it
into a formation slot around the person's estimated position.

**Two points where the code diverges from `README.md`'s description** — worth flagging in a
report rather than repeating the README's wording:

- The README calls the estimator an "Interacting Multiple Model" framework. The code's own
  docstring in [`localization_system.py`](ros2_ws/src/localization/localization/localization_system.py)
  says otherwise: it implements **IMDCL — Interim Master Decentralized Cooperative
  Localization**, from Kia, Rounds & Martínez, *"Cooperative Localization for Mobile Agents"*,
  IEEE Control Systems Magazine (the PDF at the repo root). No IMM/multiple-model logic exists
  in the code.
- The README says human tracking uses **YOLO**. In
  [`Vision_class.py`](ros2_ws/src/vision/vision/Vision_class.py) the YOLO import and model load
  are both commented out (`#from ultralytics import YOLO`, `#self.model = YOLO(...)`); the
  person is detected the same way as the robots, via an ArUco marker
  (`aruco_size_target = 0.144`, dictionary `DICT_6X6_50`).

## 2. Package map

| Package | Type | Contains |
|---|---|---|
| `vision` | ament_python | `vision_node` — ArUco detection → measurements |
| `limo_control` | ament_python | `MPC_node`, `measurement_router` |
| `localization` | ament_python | `EKF_node`, the IMDCL `EKF` class, agent motion models |
| `limo_description` | ament_python | `conf_limo.py` — shared config/constants, no nodes |
| `plots` | ament_python | `EKF_plot_node` — live visualization, no publications |
| `simulation` | ament_python | 4 standalone test-substitute nodes (§6) |
| `project_interfaces` | interfaces | custom `.msg` definitions (§4) |

No launch files exist in the repo (`find . -name "*.launch.py"` returns nothing) — nodes are
started individually, namespaced by hand at the CLI (see `ros2_commands.md` for the real
per-robot launch commands: `orbbec_camera`, `limo_bringup`, then this repo's nodes with
`--ros-args -r __ns:=/limo_N`).

## 3. Node roster

| Node | Package · executable | Instances | CLI args | Role |
|---|---|---|---|---|
| `vision_node` | `vision` | ×3 (one per robot) | `id` (`sys.argv[1]`) | ArUco detection on `/limo_<id>/color/image_raw`, emits relative measurements |
| `measurement_router` | `limo_control` | ×1 | — | Buffers latest measurement per (observer, observed) pair, re-emits one per tick, round-robin |
| `EKF_node` | `localization` | ×3 robots + ×1 person | `agent_id` positional, or `--person` | IMDCL filter for one agent. ROS node name is `extended_kalman_filter` (robots) or `person_ekf_node` (person) — **not unique across the 3 robot instances**, so they must run in separate namespaces or separate `ROS_DOMAIN_ID`s to coexist |
| `MPC_node` | `limo_control` | ×3 (one per robot) | `id` (`sys.argv[1]`) | Computes velocity command + this robot's formation slot; reads `/admin` to start/stop |
| `EKF_plot_node` | `plots` | ×1 | — | Matplotlib visualization, subscriber-only |
| `sim_camera_node` / `sim_vision_node` / `sim_odometry_node` / `sim_EKF_node` | `simulation` | ×1 each | — | Standalone stand-ins for testing without hardware, see §6 |

**Stale entry point:** `ros2_ws/src/limo_control/setup.py` still lists a console-script
`EKF_node = limo_control.EKF_node:main`, but that file no longer exists in `limo_control` —
the real `EKF_node` moved to the `localization` package. `ros2 run limo_control EKF_node`
would fail; looks safe to delete from `setup.py`.

## 4. Topics

Topics written without a leading `/` are relative — they only land in the right place
(`/limo_0/cmd_vel` etc.) if the node is launched inside a namespace, e.g.
`--ros-args -r __ns:=/limo_0`.

| Topic | Type | Publishers | Subscribers | Notes |
|---|---|---|---|---|
| `/limo_N/color/image_raw` | `sensor_msgs/Image` | camera driver (`orbbec_camera`, external to this repo) | `vision_node` ×3 | |
| `/measurement_raw` | `project_interfaces/Measurement` | `vision_node` ×3, or `sim_vision_node` | `measurement_router` | `id_a` = observer, `id_b` = observed |
| `/measurement_routed` | `project_interfaces/Measurement` | `measurement_router` | `EKF_node` ×4 | One per tick, round-robin over the 9 possible `(id_a, id_b)` pairs, rate `conf_limo.Tm` |
| `odom` → `/limo_N/odom` | `nav_msgs/Odometry` | base driver (`limo_bringup`, external) | `EKF_node` ×3 (robots only) | `sim_odometry_node`'s `/odom` is `std_msgs/Float64MultiArray` — **type mismatch**, can't substitute directly |
| `/info` | `project_interfaces/Landmark` | `EKF_node` ×4 | `EKF_node` ×4 | Peer mesh — every instance both publishes and subscribes, filters by `id_a`/`id_b` |
| `/update` | `project_interfaces/Update` | `EKF_node` ×4 | `EKF_node` ×4 | Same mesh pattern as `/info` |
| `/limo_state` | `project_interfaces/State` | `EKF_node` ×3 robots, or `sim_EKF_node` | `MPC_node` ×3, `EKF_plot_node` | |
| `/person_state` | `project_interfaces/State` | `EKF_node --person`, or `sim_EKF_node` | `MPC_node` ×3, `EKF_plot_node` | |
| `/admin` | `std_msgs/String` | external / manual (`ros2 topic pub`) — no publisher exists in the repo | `MPC_node` ×3 | QoS: RELIABLE + TRANSIENT_LOCAL, so a command sent before a node starts is still delivered; payload `'start_mpc'` / `'stop_mpc'` |
| `cmd_vel` → `/limo_N/cmd_vel` | `geometry_msgs/Twist` | `MPC_node` ×3 | base driver (external) | |
| `mpc_prediction` → `/limo_N/mpc_prediction` | `project_interfaces/MPCprediction` | `MPC_node` ×3 — **publisher exists but the `.publish()` call is commented out** in `MPC_node.py` | `EKF_plot_node` (subscribes `/limo_0/mpc_prediction` only) | Currently dead — nothing is ever published |
| `desired` → `/limo_N/desired` | `project_interfaces/Desired` | `MPC_node` ×3 | `EKF_plot_node` (subscribes `/limo_0/desired` only) | |
| `/camera/color/image_raw` | `sensor_msgs/Image` | `sim_camera_node` | — | Sim-only, absolute topic name doesn't match the namespaced production one |

`ros2_commands.md` also mentions publishing `'start_ekf'`/`'stop_ekf'` on `/admin` to control
`EKF_node`, but `EKF_node.py` has no `/admin` subscription or `admin_callback` — only
`MPC_node` reacts to `/admin`. Worth confirming with whoever wrote that note whether it's
aspirational or refers to an older version of the code.

## 5. Custom messages (`project_interfaces/msg/`)

| Message | Fields | Used for |
|---|---|---|
| `Measurement` | `id_a, id_b, x, y, dtheta` | Relative pose observation: agent `id_a` observed agent `id_b` at `(x, y, dtheta)` |
| `Landmark` | `id_a, id_b, dim, state[], phi[], p[]` | IMDCL "info" reply: `id_b`'s current state, transition Jacobian (`phi`, flattened `dim×dim`) and covariance (`p`, flattened `dim×dim`), addressed to `id_a` |
| `Update` | `id_a, id_b, dim_a, dim_b, ra[], gamma_a[], gamma_b[], w1[], w2[]` | The correction terms IMDCL broadcasts after a measurement, so every other agent can apply the same update to its own cross-covariance bookkeeping |
| `State` | `id, x, y, theta` | Fused pose estimate, broadcast periodically |
| `Desired` | `x0, y0, x1, y1, x2, y2` | The three formation slot positions computed by whichever `MPC_node` publishes it |
| `MPCprediction` | `x[], y[], theta[]` | Predicted state trajectory over the MPC horizon (currently unused, see §4) |

## 6. Distributed localization flow (why `/info` and `/update` exist)

Each `EKF_node` (robot or person) keeps its own local state + covariance and also tracks
cross-covariances with the other agents (`EKF.agent_dims`, `_cross_cov_set()`). The sequence
for one relative measurement:

1. `vision_node` on robot `A` sees agent `B`'s marker → publishes `Measurement(id_a=A, id_b=B)`
   on `/measurement_raw`.
2. `measurement_router` buffers it and eventually replays it on `/measurement_routed`.
3. Every `EKF_node` receives it. The instance with `agent_id == id_a` (the observer, `A`)
   stores it as a pending measurement keyed by `id_b`. The instance with `agent_id == id_b`
   (the observed, `B`) responds by publishing its own current state/Jacobian/covariance as a
   `Landmark` on `/info`, addressed to `id_a`.
4. `A`'s `EKF_node` picks up that `/info` reply, combines it with the pending measurement,
   runs the IMDCL update step, and publishes the resulting correction terms as an `Update` on
   `/update` (addressed `id_a → id_b`).
5. Every other agent's `EKF_node` (including `B`) receives that `/update` and applies the same
   correction to its own bookkeeping, keeping cross-covariances consistent across the team —
   this is the "cooperative" part of the estimator.

`PERSON_ID = 3` is hardcoded; the person is always `id_b` (measured), never `id_a` (measuring)
— it has no camera of its own, so the `id_a`-side branches in `EKF_node` never trigger when
running as the person.

## 7. Key constants (`limo_description/conf_limo.py`)

| Constant | Value | Meaning |
|---|---|---|
| `dt`, `Tp` | 0.1 s | EKF prediction step / state-publish period |
| `dt_MPC`, `N` | 0.1 s, 50 steps | MPC timestep and horizon (5 s lookahead) |
| `Tm` | 0.1 s | `measurement_router` replay period |
| `R_rr`, `R_rp` | `I·0.5` (3×3 / 2×2) | Measurement noise: robot↔robot vs robot↔person |
| `Q` | `I·0.001` (3×3) | Robot process noise |
| `Q_p` | constant-velocity noise model (4×4) | Person process noise |
| `r_circle`, `dist` | 0.5 m, 0.2 m | Formation radius around the person / offset used in `desired_pos()` |
| `v_max`, `w_max` | 0.1477 m/s, 0.2954 rad/s | LIMO datasheet limits used as MPC bounds |
| `aruco_size`, `aruco_size_target` | 0.060 m, 0.144 m | Physical marker sizes for pose estimation (robots vs person) |

State dimensions: robot = `[x, y, θ]` (3), person = `[x, y, vx, vy]` (4, constant-velocity
model in `PersonModel`).

## 8. Simulation substitutes (`simulation` package)

Standalone nodes for testing without hardware. None are launched together with the real nodes
they replace, and topic names/types aren't all drop-in compatible:

| Node | Publishes | Stands in for | Compatible as-is? |
|---|---|---|---|
| `sim_camera_node` | `/camera/color/image_raw` (`Image`), reading a local `.mp4` | camera driver | **No** — fixed absolute topic, not the namespaced `/limo_N/color/image_raw` `vision_node` expects |
| `sim_vision_node` | `/measurement_raw` (`Measurement`) | camera + `vision_node` together | Yes — same topic name |
| `sim_odometry_node` | `/odom` (`std_msgs/Float64MultiArray`) | base/odometry driver | **No** — `EKF_node` expects `nav_msgs/Odometry` on `odom` |
| `sim_EKF_node` | `/limo_state`, `/person_state` (`State`) | the whole `EKF_node` stack | Yes — same topic names `MPC_node`/`EKF_plot_node` expect |

## 9. Real hardware notes (from `ros2_commands.md`)

- 3 physical LIMOs, static IPs `192.168.1.37/40/34`, SSH user `agilex`.
- Camera driver: `ros2 launch orbbec_camera dabai.launch.py` (namespaced per robot). Odometry:
  `ros2 launch limo_bringup limo_start.launch.py`.
  This confirms the external "camera driver" / "odometry driver" boxes in §4 are
  `orbbec_camera` and `limo_bringup`, not part of this repo.
  Both driver launches make the confirmed image topic naming `/limo_N/color/image_raw`.
- All robots must share `ROS_DOMAIN_ID` with the operator machine (checked via
  `echo $ROS_DOMAIN_ID` on the LIMO).
- Per-robot launcher script: `./limo.sh` on each LIMO, `ros2 launch launch_multiple.py` on the
  main computer (that launch file isn't present in `ros2_ws/src` — check the repo root /
  `launch.sh` / `limo.sh` for where it actually lives).
- Deployment is via a Docker image (`ros2_ids_full`) saved/loaded across machines.

## 10. Where to look for more detail

| Question | File |
|---|---|
| Exact IMDCL update-step math | `ros2_ws/src/localization/localization/localization_system.py` |
| Motion models (robot unicycle, person CV) | `ros2_ws/src/localization/localization/agent_type.py` |
| Measurement model / ArUco → relative pose | `ros2_ws/src/localization/localization/meas_model.py`, `ros2_ws/src/vision/vision/Vision_class.py` |
| MPC cost/constraints | `ros2_ws/src/limo_control/limo_control/MPC_class.py` |
| Formation slot assignment (Hungarian algorithm) | `ros2_ws/src/limo_control/limo_control/MPC_node.py` (`desired_pos`, uses `scipy.optimize.linear_sum_assignment`) |
| CSV logging (per-agent EKF traces) | `EKF_node.py` writes `pred_states_<id>.csv` / `upd_states_<id>.csv` to `csv_data/` at the workspace root |
| Full visual diagram of §4 | published artifact "Limo Swarm Wiring" from this conversation |
