name=$(basename $1 .h5)
h5dump -b -d /data/data -o ${name}.slab ${name}.h5
h5dump -b -d /processing/cheetah/pixelmasks -o ${name}.maskslab ${name}.h5
./deslabify_mask ${name}.maskslab < geometry2.geom > ${name}.mask
./deslabify_data ${name}.slab     < geometry2.geom > ${name}.bin
