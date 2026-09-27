# 持续优化与全景基准实测推进日志 (3小时+ 持续运行)
**最后更新时间**: `2026-09-26 19:48:57`

| 时间戳 | 运行阶段 | 状态 | 详情摘要 |
|---|---|---|---|
| 2026-09-26 17:27:23 | **Campaign_Init** | `INITIALIZED` | {"target_duration": "3+ hours", "device": "NVIDIA RTX 5090"} |
| 2026-09-26 17:27:23 | **Phase1_DeepPoseTraining** | `STARTED` | {"command": "C:\\Users\\lry\\.conda\\envs\\vggt\\python.e... |
| 2026-09-26 19:42:23 | **Phase1_DeepPoseTraining** | `COMPLETED` | {"elapsed_s": 8099.3, "output_preview": ": 0.0217 | LR: 0... |
| 2026-09-26 19:42:23 | **Phase2_17SequenceBenchmark** | `STARTED` | {"command": "C:\\Users\\lry\\.conda\\envs\\vggt\\python.e... |
| 2026-09-26 19:48:44 | **Phase2_17SequenceBenchmark** | `COMPLETED` | {"elapsed_s": 381.9, "output_preview": "ity: 0.0000)\n  B... |
| 2026-09-26 19:48:44 | **Phase3_PGO_Refine_huber2.0_wt0.0001** | `RUNNING_SWEEP` | {"huber": 2.0, "reg_wt": 0.0001} |
| 2026-09-26 19:48:46 | **Phase3_PGO_Refine_huber2.0_wt1e-05** | `RUNNING_SWEEP` | {"huber": 2.0, "reg_wt": 1e-05} |
| 2026-09-26 19:48:48 | **Phase3_PGO_Refine_huber3.0_wt0.0001** | `RUNNING_SWEEP` | {"huber": 3.0, "reg_wt": 0.0001} |
| 2026-09-26 19:48:51 | **Phase3_PGO_Refine_huber3.0_wt1e-05** | `RUNNING_SWEEP` | {"huber": 3.0, "reg_wt": 1e-05} |
| 2026-09-26 19:48:53 | **Phase3_PGO_Refine_huber4.0_wt0.0001** | `RUNNING_SWEEP` | {"huber": 4.0, "reg_wt": 0.0001} |
| 2026-09-26 19:48:55 | **Phase3_PGO_Refine_huber4.0_wt1e-05** | `RUNNING_SWEEP` | {"huber": 4.0, "reg_wt": 1e-05} |
| 2026-09-26 19:48:57 | **Campaign_Summary** | `ALL_STAGES_COMPLETED_SUCCESSFULLY` | {"report_file": "E:\\MIS_TMI_Re_3D\\re3d_cmpb_results\\re... |