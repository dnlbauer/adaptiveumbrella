#!/bin/bash

lambda1=$1
lambda2=$2
FOLDER="./tmp/simulations/sim_${lambda1}_${lambda2}"


if [ -f $FOLDER/topol.gro ]; then
	echo "$FOLDER exists"
	exit 0
fi

mkdir -p $FOLDER
cp data/plumed.dat $FOLDER
cp data/topol.tpr $FOLDER
sed -i -e "s/LAMBDA1/${lambda1}/g" $FOLDER/plumed.dat
sed -i -e "s/LAMBDA2/${lambda2}/g" $FOLDER/plumed.dat

cd $FOLDER
source /usr/local/gromacs/bin/GMXRC
gmx --quiet mdrun -deffnm topol -plumed plumed.dat -nsteps 250000 2>&1
cd ../..

