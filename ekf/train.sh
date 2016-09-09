/nethome/ashaban6/caffe/build/tools/caffe train \
    -solver=solver.prototxt \
    -gpu 3 \
    2>&1 | tee ./log/real_flow128_dof4"log$(date +'%m_%d_%y')".log
