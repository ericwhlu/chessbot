# Rendered-image benchmark

All predictions were computed from decoded PNG pixels and camera calibration before comparison with ground truth.

| Case | Correct squares | Warnings |
| --- | ---: | ---: |
| initial | 64/64 | 0 |
| after_e2e4 | 64/64 | 0 |
| after_e7e5 | 64/64 | 0 |
| after_g1f3 | 64/64 | 0 |
| after_b8c6 | 64/64 | 0 |
| capture | 64/64 | 0 |
| empty | 64/64 | 0 |
| all_types_both_colors_and_tiles | 64/64 | 0 |
| arbitrary_0 | 64/64 | 3 |
| arbitrary_1 | 64/64 | 0 |
| arbitrary_2 | 64/64 | 0 |
| castling | 64/64 | 0 |
| en_passant | 64/64 | 0 |
| promotion | 64/64 | 0 |
| rotation_23 | 64/64 | 0 |
| rotation_137 | 64/64 | 0 |
| oblique_18 | 64/64 | 0 |
| oblique_30 | 64/64 | 2 |
| camera_azimuth | 64/64 | 0 |
| light_dim | 64/64 | 0 |
| light_bright | 64/64 | 0 |
| palette_slate | 64/64 | 0 |
| palette_clay | 64/64 | 0 |
| noise_4 | 64/64 | 0 |
| no_shadows | 64/64 | 0 |
| combined | 64/64 | 3 |
| stress_60_degrees | 47/64 | 14 |
| stress_dim_075 | 48/64 | 0 |
| stress_noise_12 | 64/64 | 0 |

## Actual failures

- **stress_60_degrees**: b8: . → n, c8: b → p, d8: q → r, f8: b → p, g8: n → r, e6: . → n, f4: . → N, a3: . → N, b3: . → N, c3: . → N, d3: . → N, f3: N → R, g3: . → N, h3: . → N, e2: . → N, c1: B → P, f1: B → P
- **stress_dim_075**: e4: P → ., f3: N → ., a2: P → ., b2: P → ., c2: P → ., d2: P → ., f2: P → ., g2: P → ., h2: P → ., a1: R → ., b1: N → ., c1: B → ., d1: Q → ., e1: K → ., f1: B → ., h1: R → .
