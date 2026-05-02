#!/bin/bash

if [ "$1" == "--license" ]; then
	readonly LICENSE_TEXT="QLoRA-Fine-Tuning Copyright (C) 2026 Aquin Labs
This program comes with ABSOLUTELY NO WARRANTY; for details type \`./run.sh --license'.
This is free software, and you are welcome to redistribute it
under certain conditions; type \`./run.sh --license' for details."

	echo "$LICENSE_TEXT"

else
	python ./src/main.py
fi

