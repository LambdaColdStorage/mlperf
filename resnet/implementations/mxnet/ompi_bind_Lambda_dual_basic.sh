#!/bin/bash


# case "${OMPI_COMM_WORLD_LOCAL_RANK}" in
# 0)
#     exec numactl --physcpubind=0,5 --membind=0 "${@}"
#     ;;
# 1)
#     exec numactl --physcpubind=1,6 --membind=0 "${@}"
#     ;;
# 2)
#     exec numactl --physcpubind=2,7 --membind=0 "${@}"
#     ;;
# 3)
#     exec numactl --physcpubind=3,8 --membind=0 "${@}"
#     ;;
# *)
#     echo ==============================================================
#     echo "ERROR: Unknown local rank ${OMPI_COMM_WORLD_LOCAL_RANK}"
#     echo ==============================================================
#     exit 1
#     ;;
# esac


case "${OMPI_COMM_WORLD_LOCAL_RANK}" in
0)
    export OMPI_MCA_btl_openib_if_include=mlx5_0
    exec numactl --physcpubind=0-1,10-11 --membind=0 "${@}"
    ;;
1)
    export OMPI_MCA_btl_openib_if_include=mlx5_0
    exec numactl --physcpubind=2-3,12-13 --membind=0 "${@}"
    ;;
2)
    export OMPI_MCA_btl_openib_if_include=mlx5_1
    exec numactl --physcpubind=4-5,14-15 --membind=0 "${@}"
    ;;
3)
    export OMPI_MCA_btl_openib_if_include=mlx5_1
    exec numactl --physcpubind=6-7,16-17 --membind=0 "${@}"
    ;;
*)
    echo ==============================================================
    echo "ERROR: Unknown local rank ${OMPI_COMM_WORLD_LOCAL_RANK}"
    echo ==============================================================
    exit 1
    ;;
esac
