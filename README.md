# Is there a clock in a mother's blood?

Can a single blood sample tell how many weeks pregnant someone is? This repository holds the analysis behind
the write-up at https://mavimatics.github.io/pregnancy-clock/.

It uses the multiomics pregnancy data of Ghaemi et al. (2019): 17 women, sampled once per trimester and
about six weeks after birth, each sample measured seven ways (plasma proteins, immune cells, metabolites,
cell-free RNA, plasma and serum cytokines, microbiome).

## Main findings

- Plasma proteins make by far the best clock: average error **2.1 weeks** (correlation 0.90), against
  6.0 weeks for always guessing the average. Three of the seven measurement types do worse than guessing.
- The proteins it relies on are fetal and placental (AFP, PGF, GPC3, PD-L1), and the clock reads
  early pregnancy again six weeks after birth.
- Splitting samples instead of women lets a model "predict" a random per-woman number 12% better than
  guessing, purely by recognizing individuals. All results here hold out whole women.
- With 17 women, combining the seven measurement types did not beat proteins alone.

## Reproduce

1. Download `termpregnancymultiomics.zip` from https://nalab.stanford.edu/multiomics-pregnancy/ and put
   `Data.Rda` in `data/`. The data is not redistributed here.
2. Run:

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/load_data.py          # converts Data.Rda to data/multiomics.npz
.venv/bin/python src/analysis.py           # one elastic net per measurement type
.venv/bin/python src/clock_details.py plasma_somalogic
.venv/bin/python src/clock_details.py immune_system
.venv/bin/python src/extras.py             # learned combination + identity-leak demo
.venv/bin/python src/export_web.py results/pregnancy_clock.json
```

`src/progress.py` shows a live progress bar for `analysis.py` (run it as
`.venv/bin/python -u src/analysis.py > analysis.log`). Two cores are used by default; set `N_JOBS` to change it.

## Method

Elastic net (scikit-learn `ElasticNetCV`) on standardized features. Outer loop: leave one woman out (17 folds).
Inner loop: 4-fold `GroupKFold` on the training women to tune the penalty. Cell-free RNA and SomaLogic
values are log-transformed, and cell-free RNA features that are zero in most samples are dropped. Error is
reported as mean absolute error in weeks.

## Citation

Ghaemi, M. S., DiGiulio, D. B., Contrepois, K., et al. (2019). Multiomics modeling of the immunome,
transcriptome, microbiome, proteome and metabolome adaptations during human pregnancy.
*Bioinformatics*, 35(1), 95–103.
