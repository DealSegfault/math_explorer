# Pinned benchmark inputs

- `aime_1983_2024.csv`: [gneubig/aime-1983-2024](https://huggingface.co/datasets/gneubig/aime-1983-2024), revision `1f8845323b0b5994dbd74bbd0e01dc077cbcd827`, CC0-1.0. SHA-256: `32b10a4db8739a4a204e98ced0fbe315ea291684f7e9b3f074aa48e518ee7bb5`.
- `minif2f_lean4_test.jsonl`: [Yingjia-Wan/minif2f-lean4](https://github.com/Yingjia-Wan/minif2f-lean4), revision `e62da0c6d044a361884691dd1e4919fc8c582a68`. SHA-256: `de04d26b928130e2d3967cc4ad8e1bd4efe188bd79bf24e2284692011b9adce6`.

The AIME loader excludes diagram questions (`[asy]`) and one question with multiple accepted answers. Years through 2022 form the development split; 2023–24 form the evaluation split. These are historical public questions, so the evaluation split is held out from local harness tuning, not necessarily from model training. The Lean 4 port contains 244 test rows, of which 19 are commented out as errors; the loader exposes the 225 usable theorem templates. miniF2F-test is reserved for final formal-proof evaluation.
