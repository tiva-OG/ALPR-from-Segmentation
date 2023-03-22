#!/bin/bash

set -e

mkdir data/ocr -p
mkdir data/unet -p

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=drivesdk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=drivesdk" -O final_checkpoint && rm -rf /tmp/cookies.txt -P data/unet/

wget -c -N http://sergiomsilva.com/data/eccv2018/ocr/ocr-net.cfg     -P data/ocr/
wget -c -N http://sergiomsilva.com/data/eccv2018/ocr/ocr-net.names   -P data/ocr/
wget -c -N http://sergiomsilva.com/data/eccv2018/ocr/ocr-net.weights -P data/ocr/
wget -c -N http://sergiomsilva.com/data/eccv2018/ocr/ocr-net.data    -P data/ocr/

# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-JrqM15t0ra0uKMa9y4zjWHZTzpC8vsz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-JrqM15t0ra0uKMa9y4zjWHZTzpC8vsz -O final_checkpoint && rm -rf /tmp/cookies.txt" -P data/unet/