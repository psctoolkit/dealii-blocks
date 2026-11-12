./configure \
	--with-metisdir=/opt/spack/opt/spack/linux-skylake/metis-5.1.0-v3hqm2glak6jadjaj43ra5kwbb4uq3yh/ \
	--disable-openmp \
	--with-amddir=/opt/spack/opt/spack/linux-skylake/suite-sparse-7.8.3-lr3jgmqf6konmeylquiylwk6stsczcxp \
	--with-lpk=4 \
	--disable-cuda \
	--with-ccopt="-g -ggdb -fPIC -O3 -march=native -mtune=native" \
	--with-fcopt="-g -ggdb -fPIC -O3 -march=native -mtune=native" \
	--with-cxxopt="-g -ggdb -fPIC -O3 -march=native -mtune=native" 
