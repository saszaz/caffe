/nethome/ashaban6/caffe/build/tools/caffe train \
    --solver=solver.prototxt \
    --snapshot=./snapshot/real_flow128_dof4_iter_12511.solverstate \
    -gpu 1 \
    2>&1 | tee ./log/real_flow128_dof4"log$(date +'%m_%d_%y')".log

