#!/bin/bash

check_file()
{
    if [ ! -f "$1" ]; then
        return 0
    else
        return 1
    fi
}

check_dir()
{
    if [ ! -d "$1" ]; then
        return 0
    else
        return 1
    fi
}

# Check if darknet is compiled
check_file "darkent/libdarknet.so"
retval=$?
if [ $retval -eq 1 ]; then
    echo "Darknet is not compiled! Go to 'darknet' and 'make'!"
    exit 1
fi

unet_model='data/unet/final_checkpoint'
input_dir=''
output_dir=''

# Check # of arguments
usage() {
    echo ""
    echo " Usage:"
    echo ""
    echo "  bash $0 -i input/dir -o output/dir [-h] [-l path/to/model]:"
    echo ""
    echo "  -i  Input dir path (containing JPG or PNG images)"
    echo "  -o  Output dir path"
    echo "  -u  Path to Torch UNet segmentation model (default = $unet_model)"
    echo "  -h  Print this help information"
    exit 1
}

while getopts 'i:o:u:h' OPTION; do
    case $OPTION in
        i) input_dir=$OPTARG;;
        o) output_dir=$OPTARG;;
        u) unet_model=$OPTARG;;
        h) usage;;
    esac
done

if [ -z "$input_dir" ]; then echo "Input dir not set."; usage; exit 1; fi
if [ -z "$output_dir" ]; then echo "Output dir not set."; usage; exit 1; fi

# Check if input dir exists
check_dir $input_dir
retval=$?
if [ $retval -eq 0 ]; then echo "Input dir ($input_dir) does not exist."; exit 1; fi

# Check if output dir exists, else, create it
check_dir $output_dir
retval=$?
if [ $retval -eq 0 ]; then mkdir -p $output_dir; fi

# End if any error occur
set -e

# Segment license plates
python3 lp-segmentation.py $input_dir $output_dir $unet_model

# Recognize license plates
python3 lp-recognition.py $output_dir

# Draw output and generate list
python3 gen-outputs.py $input_dir $output_dir

# Clean files and draw output
# rm $output_dir/*_plate.jpg
# rm $output_dir/*_label.txt
# rm $output_dir/*_box.txt