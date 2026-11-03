# Cold-start morning (2026-11-03 08:57)

Common lesson range: chunks 0..74 (37000 lesson batches of 32 rows each).

| run | AUC (mean SEL x, 12 levels) | END (at Kc) | SEL flags at Kc | last chunk |
|---|---|---|---|---|
| CatoA | 783.4 | 979.3 | 0 | 104 |
| CatoB | 800.5 | 1090.9 | 8 | 84 |
| CiceroA | 790.5 | 1076.6 | 0 | 108 |
| CiceroB | 734.7 | 926.9 | 12 | 93 |
| LivyA | 719.5 | 1118.9 | 0 | 90 |
| LivyB | 738.0 | 966.0 | 0 | 74 |

Noise floor (largest within-arm seed gap): AUC 55.8, END 152.9

- Cicero vs Cato: AUC dA 7.1 dB -65.8; END dA 97.3 dB -164.0 -> **NOT DISTINGUISHABLE**
- Livy vs Cato: AUC dA -63.9 dB -62.5; END dA 139.6 dB -124.9 -> **NOT DISTINGUISHABLE**

| checkpoint | 12 training levels HELD flags / mean x | World 4 first sight 4-1..4-3 flags / mean x | 4-4 maze mean x |
|---|---|---|---|
| CatoA@ck74 | 7/600, 883 | 0/150, 520 | 138 |
| CatoA@ck104 | 3/600, 1022 | 0/150, 513 | 138 |
| CatoB@ck74 | 8/600, 1012 | 0/150, 602 | 138 |
| CatoB@ck84 | 11/600, 1033 | 0/150, 751 | 138 |
| CiceroA@ck74 | 1/600, 1054 | 0/150, 571 | 138 |
| CiceroA@ck108 | 15/600, 1054 | 0/150, 414 | 138 |
| CiceroB@ck74 | 11/600, 850 | 0/150, 412 | 138 |
| CiceroB@ck93 | 18/600, 1074 | 0/150, 595 | 138 |
| LivyA@ck74 | 3/600, 1043 | 0/150, 528 | 138 |
| LivyA@ck90 | 7/600, 1044 | 0/150, 387 | 138 |
| LivyB@ck74 | 6/600, 953 | 0/150, 344 | 138 |

| gauntlet | cleared | furthest |
|---|---|---|
| CatoA_ck104_greedy_from1-1 | - | 1-1 |
| CatoA_ck104_greedy_from2-1 | - | 2-1 |
| CatoA_ck104_greedy_from3-1 | - | 3-1 |
| CatoA_ck104_greedy_from4-1 | - | 4-1 |
| CatoA_ck104_sampled_from1-1 | - | 1-1 |
| CatoA_ck104_sampled_from2-1 | - | 2-1 |
| CatoA_ck104_sampled_from3-1 | - | 3-1 |
| CatoA_ck104_sampled_from4-1 | - | 4-1 |
| CatoB_ck84_greedy_from1-1 | - | 1-1 |
| CatoB_ck84_greedy_from2-1 | - | 2-1 |
| CatoB_ck84_greedy_from3-1 | - | 3-1 |
| CatoB_ck84_greedy_from4-1 | - | 4-1 |
| CatoB_ck84_sampled_from1-1 | - | 1-1 |
| CatoB_ck84_sampled_from2-1 | - | 2-1 |
| CatoB_ck84_sampled_from3-1 | - | 3-1 |
| CatoB_ck84_sampled_from4-1 | - | 4-1 |
| CiceroA_ck108_greedy_from1-1 | - | 1-1 |
| CiceroA_ck108_greedy_from2-1 | - | 2-1 |
| CiceroA_ck108_greedy_from3-1 | - | 3-1 |
| CiceroA_ck108_greedy_from4-1 | - | 4-1 |
| CiceroA_ck108_sampled_from1-1 | 1-1 | 1-2 |
| CiceroA_ck108_sampled_from2-1 | - | 2-1 |
| CiceroA_ck108_sampled_from3-1 | - | 3-1 |
| CiceroA_ck108_sampled_from4-1 | - | 4-1 |
| CiceroB_ck93_greedy_from1-1 | - | 1-1 |
| CiceroB_ck93_greedy_from2-1 | - | 2-1 |
| CiceroB_ck93_greedy_from3-1 | - | 3-1 |
| CiceroB_ck93_greedy_from4-1 | - | 4-1 |
| CiceroB_ck93_sampled_from1-1 | - | 1-1 |
| CiceroB_ck93_sampled_from2-1 | - | 2-1 |
| CiceroB_ck93_sampled_from3-1 | - | 3-1 |
| CiceroB_ck93_sampled_from4-1 | - | 4-1 |
| LivyA_ck90_greedy_from1-1 | - | 1-1 |
| LivyA_ck90_greedy_from2-1 | - | 2-1 |
| LivyA_ck90_greedy_from3-1 | - | 3-1 |
| LivyA_ck90_greedy_from4-1 | - | 4-1 |
| LivyA_ck90_sampled_from1-1 | - | 1-1 |
| LivyA_ck90_sampled_from2-1 | - | 2-1 |
| LivyA_ck90_sampled_from3-1 | - | 3-1 |
| LivyA_ck90_sampled_from4-1 | - | 4-1 |
| LivyB_ck74_greedy_from1-1 | - | 1-1 |
| LivyB_ck74_greedy_from2-1 | - | 2-1 |
| LivyB_ck74_greedy_from3-1 | - | 3-1 |
| LivyB_ck74_greedy_from4-1 | - | 4-1 |
| LivyB_ck74_sampled_from1-1 | - | 1-1 |
| LivyB_ck74_sampled_from2-1 | - | 2-1 |
| LivyB_ck74_sampled_from3-1 | - | 3-1 |
| LivyB_ck74_sampled_from4-1 | - | 4-1 |
