#!/bin/zsh
src=/home/usr/src 
g++ -march=native -Wfatal-errors -std=c++17 -O3 -DNDEBUG\
  -I${src}/patchmap \
  -I${src}/dlib -I${src}/geometry -I${src}/partiality/ -I${src}/asu/ \
  -I${src}/wmath/ c2g.cpp -o c2g
g++ -O3 -std=c++17 -I${src}/geometry -I${src}/wmath geom2bin.cpp -o geom2bin
g++ -march=native -Wfatal-errors -std=c++17 -O3 -DNDEBUG\
  -I${src}/patchmap \
  -I${src}/dlib -I${src}/geometry -I${src}/partiality/ -I${src}/asu/ \
  -I${src}/wmath/ ./geom2bin.cpp -o geom2bin
g++ -march=native -Wfatal-errors -std=c++17 -O3 -DNDEBUG\
  -I${src}/patchmap \
  -I${src}/dlib -I${src}/geometry -I${src}/partiality/ -I${src}/asu/ \
  -I${src}/wmath/ ./parm2bin.cpp -o parm2bin
g++ -O3 -std=c++17 -DNDEBUG\
  -I${src}/partiality/ -I${src}/patchmap/ -I${src}/wmath/ -I${src}/geometry/\
  -I${src}/dlib/ fit.cpp -o fit

h5dump -b -d /data/data -o LCLS_2013_Mar20_r0016_150851_1801b.slab LCLS_2013_Mar20_r0016_150851_1801b.h5
h5dump -b -d /processing/cheetah/pixelmasks -o LCLS_2013_Mar20_r0016_150851_1801b.maskslab LCLS_2013_Mar20_r0016_150851_1801b.h5
./deslabify_data LCLS_2013_Mar20_r0016_150851_1801b.slab     < geometry2.geom \
  > LCLS_2013_Mar20_r0016_150851_1801b.bin
./deslabify_data LCLS_2013_Mar20_r0016_150851_1801b.maskslab < geometry2.geom \
  > LCLS_2013_Mar20_r0016_150851_1801b.mask
cat <(./parm2bin $0.parm) \
    <(cat <(echo 582.00e-3) geometry2.geom | ./c2g | ./geom2bin) \
    LCLS_2013_Mar20_r0017_152236_81f.mask LCLS_2013_Mar20_r0017_152236_81f.bin \
    | ./fit \
    > fit.out \
    2> fit.log
#g++ -march=native -Wfatal-errors -std=c++17 -O3 -DNDEBUG -I/home/usr/src/partiality/ -I/home/usr/src/patchmap/ -I/home/usr/src/wmath/ -I/home/usr/src/geometry/ -I/home/usr/src/dlib/ ./fit.cpp -lboost_container -o fit && ( name="LCLS_2013_Mar20_r0041_223336_1eea"; cat <(./parm2bin < ${name}.parm) <(./geom2bin < ${name}.geom) ${name}.mask ${name}.bin | ./fit )
