# Colab evaluation farm

The local link reaches the ETH EuRoC server at about 1 MB/s. Colab reaches it at about 45 MB/s,
so multi-flight validation runs in Colab while development happens locally.

## Recreate the workspace (fresh runtime)

1. Create the folders `/content/pf/{src,tools,data/raw,data/degraded,data/out}`.
2. Copy the files in with `%%writefile` cells:
   * `src/main.py` → `/content/pf/src/main.py` (**must be the current local version**)
   * `tools/colab/fetch_all.py` → `/content/pf/fetch_all.py`
   * `tools/colab/harness.py` → `/content/pf/tools/harness.py`
   * the organiser's `degrade_imu.py` (kept outside this repo) →
     `/content/pf/tools/degrade_imu.py`
3. Download the flights in the background (stay polite: 8 workers):
   `nohup python fetch_all.py MH_01_easy ... V2_03_difficult > fetch.log &`
4. Evaluate:
   ```python
   import sys; sys.path.insert(0, '/content/pf/tools'); import harness
   harness.build_new()
   res = harness.evaluate()                      # current CFG
   harness.norm_score(res)                       # (mean, worst) vs baseline
   harness.sweep({'vel_hp_s': [6.0, 12.0], 'pos_hp_s': [10.0, 20.0]})
   ```

## Parity check

After every sync, compare MH_04 numbers between Colab and `tools/evaluate.py`.
They matched exactly on 21 Nov (ordered 5.69 m, literal 5.90 m at that
commit).
