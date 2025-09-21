# Keyboard Temperament Estimation from Symbolic Data

This repository contains supplemental material for the paper:

**Peter van Kranenburg and Gerben Bisschop (2025). Keyboard Temperament Estimation from Symbolic Data: A Case Study on Bach's Well-Tempered Clavier. The 26th International Society for Music Information Retrieval Conference, Daejeon.**

## Overview

This project provides a data-driven approach to estimate optimal keyboard temperaments from symbolic music data, focusing on minimizing deviations from pure intervals. The method is applied to Johann Sebastian Bach's *Well-Tempered Clavier* (WTC) as a case study. The repository includes Python code, data, and a web application to explore the results.

## Repository Contents

- **code/**: Python scripts and notebooks implementing the temperament estimation method, including:
  - temperament_functions.py: library of functions for analysis, estimation and representation.
  - temperament_major.ipynb: notebook with the code for the C-major corpus
  - temperament_uniform.ipynb: notebook with the code for the uniformely distributed corpus.
  - temperament_wtc.ipynb: notebook with the code for Bach's Wohltemperierte Clavier.
- **wtc-examples/**: Synthesized midi files for the entire Wohltemperierte Clavier, including:
  - wtc_equal: Equal temperament.
  - wtc_meantone: 1/4 comma meantone temperament.
  - wtc_kelletat1: Herbert Kelletat's Bach temperament.
  - wtc_ramis: Temperament by Ramis de Pareia (1482).
  - wtc_optimal: Temperament found by our method.
  - wtc_optimal_bounded_fifths: Temperament found by our method, with all fifhts bounded between 696 and 705 cents.
  - wtc_optimal_just_targets: Temperament found by our method, with only the 5-limit just intervals as acceptable targets.
- **results/**: plots and json files containing the output of our method.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{vankranenburg2025,
  author = {van Kranenburg, Peter and Bisschop, Gerben},
  title = {Keyboard Temperament Estimation from Symbolic Data: A Case Study on Bach's Well-Tempered Clavier},
  booktitle = {Proceedings of the 26th International Society for Music Information Retrieval Conference},
  year = {2025},
  address = {Daejeon}
}
```

## Contact

For questions or contributions, contact:
- Peter van Kranenburg: p.vankranenburg@uu.nl
- Gerben Bisschop: g.bisschop@uu.nl