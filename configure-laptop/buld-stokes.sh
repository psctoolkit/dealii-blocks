DEAL_II_DIR_LOCAL=$(cd build & pwd)

echo "Selected DEAL_II_DIR is ${DEAL_II_DIR_LOCAL}"

cmake -S . -B build-stokes \
  -DDEAL_II_DIR=${DEAL_II_DIR_LOCAL} \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-stokes -j