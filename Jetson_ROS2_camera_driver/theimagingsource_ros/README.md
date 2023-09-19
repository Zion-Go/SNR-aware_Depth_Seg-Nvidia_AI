# theimagingsource_ros

### I little note about folder and script purposes:
- **src** - a folder for source code or python scripts (ros node, for example);
- **inlude** - a folder for header files (if you program in c++);
- **docker** - a folder for all docker-related files: Dockerfile, docer-compose files, env files, etc.;
- **docker/scripts/build.sh** - a script that should build the docker image;
- **docker/scripts/up.sh** - a script that should run a docker container from the built docker image;
- **config** - a folder for config files (.yaml) with parameters, needed for setting the ROS node (camera serial number, resolution, etc.). This folder should by binded to the docker container, so you can apply files inside docker;
- **launch** - a folder for launch files (.launch), that start the ROS node and load parameters from configuration files described above. This folder should by binded to the docker container, so you can run files inside docker;

P.S. Do not forget about adding CmakeLists.txt and package.xml to the root of the repo, as it is a ROS repo.
