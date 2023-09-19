#!/bin/bash

cd "$(dirname "$0")"
cd ../tis_repos

git clone https://github.com/TheImagingSource/tiscamera.git --branch v-tiscamera-1.0.0
git clone https://github.com/TheImagingSource/Linux-tiscamera-Programming-Samples --branch master
