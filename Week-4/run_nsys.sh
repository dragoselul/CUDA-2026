#!/bin/bash

# --- 1. Setup the Environment ---
# Ensure your Arch shell knows where the tools are on the GH200
module load nvhpc/25.9

# --- 2. Configuration ---
APP_NAME="./reduction"
REPORT_NAME="reduction_gh200_report"

# --- 3. The nsys command ---
# --trace: We track CUDA, OS Runtime, and NVLink (vital for GH200)
# --sample=cpu: Watches what the Grace CPU is doing during the "gaps"
# --stats=true: Dumps a summary table directly in your terminal
nsys profile \
    --trace=cuda,osrt,nvtx \
    --sample=cpu \
    --output=$REPORT_NAME \
    --force-overwrite=true \
    --stats=true \
    $APP_NAME

echo "------------------------------------------------"
echo "Profile Complete!"
echo "To view: scp [remote]:$(pwd)/$REPORT_NAME.nsys-rep ."
echo "Then run: nsys-ui $REPORT_NAME.nsys-rep on your Arch desktop."