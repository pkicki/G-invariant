for N in {1..10}; do
  python invariance_area.py --config-file ./config_files/area.conf --n=32 --model=FC_G-inv --out-name area_nmid/my_inv_fc_32_$N &
  python invariance_area.py --config-file ./config_files/area.conf --n=32 --model=Conv1D_G-inv --out-name area_nmid/my_inv_conv_32_$N &
done
