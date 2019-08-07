ao-beam -2016 -allsky -proto 1147313992_121-132_proto.fits -m 1147313992.metafits
../lookup_jones.py -f 161.9 1147313992 _121-132_proto.fits beam- _lookup.fits --beam_path=/data/other/pb_lookup/gleam_jones.hdf5
