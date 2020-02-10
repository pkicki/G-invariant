for N in {1..10}; do
#python invariance_area.py --config-file ./config_files/area_nmid.conf --out-name area_nmid/my_inv_conv_64_$N;
#python invariance_polyGH.py --config-file ./config_files/polyGH.conf --out-name poly_GH_2d16/Z3_$N &
#python invariance_polyGH2.py --config-file ./config_files/polyGH.conf --out-name poly_GH2/S3xS2_$N;
#python invariance_polyGH4.py --config-file ./config_files/polyGH.conf --out-name poly_GH2/Z3_$N;
#python invariance_poly_Z5.py --config-file ./config_files/poly.conf --out-name poly_Z5/maron_$N &
#python invariance_poly_D8.py --config-file ./config_files/poly.conf --out-name poly_D8/maron_$N &
python invariance_poly_S4.py --config-file ./config_files/poly.conf --out-name poly_S4/my_inv_fc_$N &
done
