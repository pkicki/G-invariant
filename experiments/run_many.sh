for N in {1..10}; do
python invariance_area.py --config-file ./config_files/area.conf --out-name area/maron_imp_$N;
#python invariance_area_img.py --config-file ./config_files/area.conf --out-name area/conv_img_$N;
#python invariance_poly.py --config-file ./config_files/poly.conf --out-name poly/conv_my_inv_$N;
done
