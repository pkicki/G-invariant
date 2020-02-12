for N in {1..10}; do
  python invariance_area.py --config-file ./config_files/area.conf --n=2 --model=Conv1D_G-avg --out-name area4/avg_conv_$N &
done
