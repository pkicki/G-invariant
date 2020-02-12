for N in {1..10}; do
  python invariance_poly_Z5.py --config-file ./config_files/poly.conf --model=Maron --out-name poly/maron_$N &
done
