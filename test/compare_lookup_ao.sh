#ao-beam -2016 -allsky -name 0000000000 -delays 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
#gen_fake_metafits.py
ln -s 0000000000-xxr.fits 0000000000.fits
../lookup_jones.py -f 149.76 0000000000 .fits 0000000000 _lookup.fits --beam_path=/data/other/pb_lookup/gleam_jones.hdf5
