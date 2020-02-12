for N in {1..10}; do
  python invariance_poly_groups.py --config-file ./config_files/poly_groups.conf --group=A4 --poly=A4 --out-name poly_groups/A4_$N &
done
