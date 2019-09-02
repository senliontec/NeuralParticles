#!/bin/sh

for file in $1
do
	if [ -d "$file$2" ] && [ -z "$(ls -A $file$2)" ]; then
		echo $file
		rm -r $file
	fi
done

