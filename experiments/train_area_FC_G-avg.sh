for N in {1..10}; do
  python invariance_area.py --config-file ./config_files/area.conf --n=2 --model=FC_G-avg --out-name area4/avg_fc_$N &
done
