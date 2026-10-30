# Complete Track 3 submission — 2026-10-30

All six Kaggle submissions completed scoring successfully. Account: Bap3r_681.

20 episodes, 36 reconstructed objects, 9,120 actual object-pose rows. Native 20 Hz, fixed world0; object assistance zero at every control step. Existing learned actors, zero new training. All failures retained.

| Metric | Official score | Displayed rank |
|---|---:|---:|
| [AUC](https://www.kaggle.com/competitions/v2d-challenge-track3-auc/leaderboard) | 0.15786 | 33/35 |
| [SP-SR](https://www.kaggle.com/competitions/v2d-challenge-track3-sp-sr/leaderboard) | 0.05000 | 31/34 |
| [MP-SR](https://www.kaggle.com/competitions/v2d-challenge-track3-mp-sr/leaderboard) | 0.00000 | 34/34 |
| [RPE](https://www.kaggle.com/competitions/v2d-challenge-track3-rpe/leaderboard) | 25.01154 | 31/34 |
| [MPPE](https://www.kaggle.com/competitions/v2d-challenge-track3-mppe/leaderboard) | 20.82605 | 30/34 |
| [CD-O](https://www.kaggle.com/competitions/v2d-challenge-track3-cd-o/leaderboard) | 0.51105 | 25/34 |

RPE, MPPE and CD-O are centimetres. Rankings are a time-specific UI snapshot; tied scores can occupy different displayed positions. These are hidden-reference official scores, not local CHORD or reconstructed-reference proxies.

Public producing code: https://github.com/BAp3r/v2d-track3-submission/commit/c3e975141a3a5a5c497ecdbdf9fd99fe6b7facb7

Artifacts: `.e2e/zero_success_20261027/full-submission-20261030-001/` contains exact pack hashes, source/checkpoint manifests, actual actions/states/contacts, all20 SwanLab/TensorBoard verified metric series, Kaggle page evidence and screenshots. `submission-readiness.json` and `complete-evidence.json` hold the final receipt. Kaggle did not expose a numeric submission ID in the observed UI; no ID is invented.

Quality remains limited: local CHORD0/20 and official MP-SR0; this round establishes valid complete delivery. Natural grasp/contact refinements did not block delivery.
