for N in {1..10}; do
  python invariance_area.py --config-file ./config_files/area.conf --n=2 --model=Conv1D_G-inv --out-name area4/my_inv_conv_$N &
done
