PSBLASDIR=$(cd ../psblas3 && pwd)

./configure --with-psblas=${PSBLASDIR} \
	--with-mumpsincdir=/opt/spack/opt/spack/linux-skylake/mumps-5.8.2-ykrl3js5mjqkofzsb6eexnkg6pfttgtr/include \
	--with-mumpslibdir=/opt/spack/opt/spack/linux-skylake/mumps-5.8.2-ykrl3js5mjqkofzsb6eexnkg6pfttgtr/lib \
	--with-mumps="-L/opt/spack/opt/spack/linux-skylake/mumps-5.8.2-ykrl3js5mjqkofzsb6eexnkg6pfttgtr/lib -lsmumps -ldmumps -lcmumps -lzmumps -lmumps_common -lpord -lscalapack -L/opt/spack/opt/spack/linux-skylake/netlib-scalapack-2.2.2-nacqpzdatxbxq66qx6baxeyc4anf2l72/lib -lopenblas -lmetis -fopenmp" \
	--with-ccopt="-g -ggdb -fPIC -O3 -march=native -mtune=native" \
	--with-fcopt="-g -ggdb -fPIC -O3 -march=native -mtune=native" \
	--with-cxxopt="-g -ggdb -fPIC -O3 -march=native -mtune=native" 

