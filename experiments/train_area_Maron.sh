for N in {1..10}; do
  python invariance_area.py --config-file ./config_files/area.conf --n=2 --model=Maron --out-name area4/maron_$N &
done
