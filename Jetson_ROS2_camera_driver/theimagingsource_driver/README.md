# theimagingsource_driver

This repo is designed to use TheImagingSource cameras from docker (**Ubuntu focal**) on **x86_64/arm64**.

## Prerequisites

- [git](https://github.com/git-guides/install-git)
- [docker](https://docs.docker.com/engine/install/ubuntu/)
- [docker-compose](https://docs.docker.com/compose/install/)


## Installation

Clone the repo, go to the repo folder:
```
git clone https://gitlab.com/sk-isrl/sdc-kia/stereo/theimagingsource_driver.git --branch main
cd theimagingsource_driver
```
Clone tis repos (examples):
```
./scripts/git_clone.sh
```
Build the docker image:
```
./docker/scripts/build.sh
```

## Launch
Run the docker container from the built image:
```
./docker/scripts/up.sh
```
To go inside the docker, execute:
```
./docker/scripts/into.sh
```