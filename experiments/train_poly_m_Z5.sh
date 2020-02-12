for N in {1..10}; do
  python invariance_poly_groups.py --config-file ./config_files/poly_groups.conf --group=Z5 --poly=Z5 --out-name poly_groups/Z5_$N &
done
