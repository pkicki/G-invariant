for N in {1..10}; do
  python invariance_area.py --config-file ./config_files/area.conf --n=2 --model=FC_G-inv --out-name area_nmid/my_inv_fc_2_$N &
  python invariance_area.py --config-file ./config_files/area.conf --n=2 --model=Conv1D_G-inv --out-name area_nmid/my_inv_conv_2_$N &
done
