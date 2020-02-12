for N in {1..10}; do
  python invariance_poly_groups.py --config-file ./config_files/poly_groups.conf --n=8 --ts=10 --group=S3 --poly=S3 --out-name poly_GH/S3_$N &
done
