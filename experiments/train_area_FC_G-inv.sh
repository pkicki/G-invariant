for N in {1..10}; do
  python invariance_area.py --config-file ./config_files/area.conf --n=2 --model=FC_G-inv --out-name area4/my_inv_fc_$N &
done
