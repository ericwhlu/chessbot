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
| arbitrary_0 | 64/64 | 0 |
| arbitrary_1 | 64/64 | 0 |
| arbitrary_2 | 64/64 | 0 |
| castling | 64/64 | 0 |
| en_passant | 64/64 | 0 |
| promotion | 64/64 | 0 |
| rotation_23 | 64/64 | 0 |
| rotation_137 | 64/64 | 0 |
| oblique_18 | 64/64 | 0 |
| oblique_30 | 64/64 | 0 |
| camera_azimuth | 64/64 | 0 |
| light_dim | 64/64 | 0 |
| light_bright | 64/64 | 0 |
| palette_slate | 64/64 | 0 |
| palette_clay | 64/64 | 0 |
| noise_4 | 64/64 | 0 |
| no_shadows | 64/64 | 0 |
| combined | 64/64 | 10 |
| stress_60_degrees | 43/64 | 28 |
| stress_dim_075 | 48/64 | 0 |
| stress_noise_12 | 64/64 | 0 |

## Actual failures

- **stress_60_degrees**: b8: . → r, c8: b → r, f8: b → r, a7: p → r, c7: p → r, e6: . → r, e5: p → r, f4: . → R, a3: . → R, b3: . → R, c3: . → R, d3: . → R, g3: . → R, h3: . → R, a2: P → R, b2: P → R, c2: P → R, d2: P → R, e2: . → P, f2: P → R, h2: P → R
- **stress_dim_075**: e4: P → ., f3: N → ., a2: P → ., b2: P → ., c2: P → ., d2: P → ., f2: P → ., g2: P → ., h2: P → ., a1: R → ., b1: N → ., c1: B → ., d1: Q → ., e1: K → ., f1: B → ., h1: R → .
