#!/usr/bin/env bash
# This script is used to record training data for a speaker

if [[ -z ${1} ]]; then
	echo 'Please specify the data directory as the first argument.'
	exit 1
fi

DATA_DIR=${1}
saved_dir=$(pwd)

echo -n 'Enter the name of the Speaker: ' 
read speaker
echo -n 'Enter the number of files to record: '
read fcount 

DEST_DIR="${DATA_DIR}/${speaker}"
mkdir ${DEST_DIR}
cd ${DEST_DIR}

for i in $(seq 1 ${fcount}); do
	echo -n "Press any key to start recording."
	read tmp
	echo "Recording file #${i}..."
	arecord -d 5s -t wav -r 48000 "clip${i}.wav"
done

cd ${saved_dir}




