for N in {1..10}; do
  python invariance_poly_Z5.py --config-file ./config_files/poly.conf --model=FC_G-inv --out-name poly/my_inv_fc_$N &
done
