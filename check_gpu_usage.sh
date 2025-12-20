#!/bin/bash
scontrol show node | awk '
/NodeName=/ {
    split($1,a,"=");
    node=a[2]
}
/Gres=/ {
    # 提取 GPU 型号，比如 a100 或 a40
    match($0,"gpu:([^:]+)",m);
    gputype=m[1]
}
/CfgTRES=/ {
    match($0,"gres/gpu=([0-9]+)",m);
    total=m[1]
}
/AllocTRES=/ {
    match($0,"gres/gpu=([0-9]+)",m);
    used=m[1]
    free=total-used
    printf "%-10s  type=%-5s  total=%2d  used=%2d  free=%2d\n", node, gputype, total, used, free
}'
