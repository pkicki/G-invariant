for N in {1..10}; do
  python invariance_poly_groups.py --config-file ./config_files/poly_groups.conf --group=D8 --poly=D8 --out-name poly_groups/D8_$N &
done
