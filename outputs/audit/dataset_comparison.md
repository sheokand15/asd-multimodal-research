# Dataset comparison

| | Move4AS | MMASD | Q-CHAT-10 toddlers (Kaggle) |
|---|---|---|---|
| Population | adults (ASD 14, control 20) | children with ASD only (32) | toddlers 12-36 months (1054 rows) |
| ASD label | clinical group (folder P*/S*) | none (all ASD) | screening outcome = Q-CHAT sum > 3 |
| Valid binary ASD vs non-ASD task | **yes** | **no** | only as rule reproduction / demographics-only |
| Modalities | 3D mocap (21 rigid bodies), EEG (15/16 ch, 250 Hz), IMU | OpenPose 2D (25), ROMP 3D (24/71), optical flow, ADOS-2 | 10 binary items + demographics |
| Participant IDs | P1-P14, S1-S20 | 5-digit IDs | Case_No (row id) |
| Clinical scores | AQ (controls), ADOS (some ASD) | ADOS-2 for 20 participants with clips | none (screen score only) |
| Licence | CC BY 4.0 | research dataset (see repo) | Kaggle / original author terms |
| Key issues | small n; task-mix and 40-fps acquisition confounds | no controls; subject identity unlabelled in multi-person frames | target leakage by construction; Combined file mislabelled |

**Cross-dataset pairing:** none of the three datasets share participants, age group or label definition.
No multimodal fusion across datasets is performed; questionnaire fusion with Move4AS/MMASD is impossible
(would require questionnaires collected from the same Move4AS participants — e.g. the AQ, which exists only for
controls and therefore cannot be used).
