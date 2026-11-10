# 5.0 closed-loop flux comparison (240 history file pairs)

| flux | p50 | p99 | max | p99 limit | snow p99 | frozen p99 | cond limit | result |
|---|---|---|---|---|---|---|---|---|
| Qle | 0.000746 | 0.0379 | 271 | 5 | 0.0354 (n=5507149) | 0.0446 (n=3162380) | 15 | pass |
| Qh | 0.00174 | 0.485 | 425 | 10 | 0.29 (n=5507149) | 2.17 (n=3162380) | 30 | pass |
| Qg | 0.000872 | 0.628 | 432 | 5 | 0.312 (n=5507149) | 2.61 (n=3162380) | 15 | pass |
| Swnet | 0 | 0.0111 | 338 | 1 | 0.0538 (n=5507149) | 0.0427 (n=3162380) | 3 | pass |
| Lwnet | 0.000304 | 0.12 | 58.1 | 2 | 0.0666 (n=5507149) | 0.535 (n=3162380) | 6 | pass |
| Evap | 2.82e-10 | 1.47e-08 | 0.000108 | 2e-06 | 1.27e-08 (n=5507149) | 1.57e-08 (n=3162380) | 6e-06 | pass |

Energy balance R = Swnet + Lwnet - Qle - Qh - Qg (sign chosen by Fortran closure).
Fortran |R| p50 0.194, p99 11.6; JAX |R| p50 0.194, p99 11.6; |R_jax - R_fortran| p99 0.0199 (limit 1): pass

Overall: PASS against thresholds fixed before the run (see script header).
