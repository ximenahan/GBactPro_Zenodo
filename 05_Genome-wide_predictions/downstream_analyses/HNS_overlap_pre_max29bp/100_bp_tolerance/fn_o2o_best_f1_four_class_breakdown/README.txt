fn_o2o_best_f1_four_class_breakdown/
  analyze_fn_four_class.py   Classify each FN TSS using union-table flags
  run.sh                     conda + python entrypoint
  fn_counts_by_tss_class.tsv (written by run)
  fn_class_bar_chart.png     (written by run)
  fn_class_bar_chart_highres.png
  run_metadata.json

Input FN list default:
  ../o2o_best_f1_tp_fn_promoter_logos/promoter_architecture_scan/input/ecoli_fn_tss_subset.csv

Class labels come from columns Primary, Secondary, Internal, Antisense on
each row (same schema as ecoli_all4classes_tss_filtered.csv). Exactly one
flag is 1 per row in the full union table.

Optional override:
  ./run.sh --fn-csv /path/to/other_fn_subset.csv
