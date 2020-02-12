for N in {1..10}; do
  python invariance_area.py --config-file ./config_files/area.conf --n=1 --model=FC_G-inv --out-name area_nmid/my_inv_fc_1_$N &
  python invariance_area.py --config-file ./config_files/area.conf --n=1 --model=Conv1D_G-inv --out-name area_nmid/my_inv_conv_1_$N &
done
