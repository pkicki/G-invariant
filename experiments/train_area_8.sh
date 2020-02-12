for N in {1..10}; do
  python invariance_area.py --config-file ./config_files/area.conf --n=8 --model=FC_G-inv --out-name area_nmid/my_inv_fc_8_$N &
  python invariance_area.py --config-file ./config_files/area.conf --n=8 --model=Conv1D_G-inv --out-name area_nmid/my_inv_conv_8_$N &
done
