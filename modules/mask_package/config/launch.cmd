#! /bin/bash

# Set path to package module
if [[ ! $PACKAGE_DIR ]]; then
	export PACKAGE_DIR="$HOME/ShapePipe/modules/mask_package"
fi

# Set path to config directory
# Can be set to non-default value, e.g. to current directory with
# export CONFIG_DIR=`pwd`
if [[ ! $CONFIG_DIR ]]; then
	export CONFIG_DIR="$PACKAGE_DIR/config"
fi

# Run package
cmd="python ${PACKAGE_DIR}/mask/mask_SMP.py -d ${CONFIG_DIR} -c package_config_smp.cfg"
$cmd
