#!/bin/bash

run_gradcam() {
    idx=$1
    branch=$2
    sequence_length=$3
    public_flag=$4

    if [ "$branch" = "bacteria" ]; then
        target_layer="conv3"
    else
        target_layer="conv2"
    fi

    cmd="python gradcam_to_blast.py \
        --idx $idx \
        --target_layer $target_layer \
        --branch $branch \
        --num_regions 3 \
        --window_size 500 \
        --gradcam_type guided"

    if [ "$public_flag" = "true" ]; then
        cmd="$cmd --public"
    fi

    cmd="$cmd $sequence_length"

    echo "Running: $cmd"
    eval $cmd
}

### --- PUBLIC DATASET --- ###
# True Positive (idx=0)
run_gradcam 0 phage 53124 true
run_gradcam 0 bacteria 6988208 true

# False Positive (idx=1994)
run_gradcam 1994 phage 54867 true
run_gradcam 1994 bacteria 6988208 true

# True Negative (idx=1993)
run_gradcam 1993 phage 41526 true
run_gradcam 1993 bacteria 6988208 true

# False Negative (idx=91)
run_gradcam 91 phage 46732 true
run_gradcam 91 bacteria 6988208 true

### --- PRIVATE DATASET --- ###
# True Positive (idx=0)
run_gradcam 0 phage 18227 false
run_gradcam 0 bacteria 2872769 false

# False Positive (idx=13)
run_gradcam 13 phage 43114 false
run_gradcam 13 bacteria 2692583 false

# True Negative (idx=10)
run_gradcam 10 phage 18227 false
run_gradcam 10 bacteria 2692583 false

# False Negative (idx=7)
run_gradcam 7 phage 41708 false
run_gradcam 7 bacteria 2774913 false
