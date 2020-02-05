for N in {1..10}; do
#python invariance_area_message_passing.py --config-file ./config_files/area.conf --out-name area7/message_passing_$N;
#python invariance_area_segment.py --config-file ./config_files/area.conf --out-name area7/segment_$N;
#python invariance_area_avg_conv.py --config-file ./config_files/area.conf --out-name area7/avg_conv_$N;
#python invariance_area_avg_fc.py --config-file ./config_files/area.conf --out-name area7/avg_fc_$N;
#python invariance_area_my_inv_conv.py --config-file ./config_files/area.conf --out-name area7/my_inv_conv_$N;
#python invariance_area_my_inv_fc.py --config-file ./config_files/area.conf --out-name area7/my_inv_fc_$N;
#python invariance_area_maron4.py --config-file ./config_files/area4.conf --out-name area4/maron_$N;
#python invariance_area_img.py --config-file ./config_files/area7.conf --out-name area7/conv_img_$N &
#python invariance_area_my_inv_conv.py --config-file ./config_files/area_nmid.conf --out-name area_nmid/my_inv_conv_64_$N;
#python invariance_polyGH.py --config-file ./config_files/polyGH.conf --out-name poly_GH_2d16/Z3_$N &
#python invariance_polyGH2.py --config-file ./config_files/polyGH.conf --out-name poly_GH2/S3xS2_$N;
#python invariance_polyGH4.py --config-file ./config_files/polyGH.conf --out-name poly_GH2/Z3_$N;
#python invariance_poly_my_inv_fc.py --config-file ./config_files/poly.conf --n 1 --out-name poly_nmid/my_inv_fc_1_$N &
#python invariance_poly_my_inv_conv.py --config-file ./config_files/poly.conf --n 1 --out-name poly_nmid/my_inv_conv_1_$N &
#python invariance_poly_Z5.py --config-file ./config_files/poly.conf --out-name poly_Z5/maron_$N &
#python invariance_poly_D8.py --config-file ./config_files/poly.conf --out-name poly_D8/maron_$N &
python invariance_poly_S4.py --config-file ./config_files/poly.conf --out-name poly_S4/my_inv_fc_$N &
#python invariance_poly_my_inv_conv_layers.py --config-file ./config_files/poly.conf --out-name poly_layers_conv/my_inv_conv_double_conv_$N &
done
