for N in {1..10}; do
  python invariance_poly_groups.py --config-file ./config_files/poly_groups.conf --group=S4 --poly=S4 --out-name poly_groups/S4_$N &
done
