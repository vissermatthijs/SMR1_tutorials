cmake_minimum_required(VERSION 3.5)
project(SMR1)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        build/catkin_tools_prebuild/catkin/catkin_generated/version/package.cmake
        build/catkin_tools_prebuild/catkin_generated/installspace/_setup_util.py
        build/catkin_tools_prebuild/catkin_generated/installspace/catkin_tools_prebuild.pc
        build/catkin_tools_prebuild/catkin_generated/installspace/catkin_tools_prebuildConfig-version.cmake
        build/catkin_tools_prebuild/catkin_generated/installspace/catkin_tools_prebuildConfig.cmake
        build/catkin_tools_prebuild/catkin_generated/installspace/env.sh
        build/catkin_tools_prebuild/catkin_generated/installspace/setup.bash
        build/catkin_tools_prebuild/catkin_generated/installspace/setup.sh
        build/catkin_tools_prebuild/catkin_generated/installspace/setup.zsh
        build/catkin_tools_prebuild/catkin_generated/stamps/catkin_tools_prebuild/_setup_util.py.stamp
        build/catkin_tools_prebuild/catkin_generated/stamps/catkin_tools_prebuild/interrogate_setup_dot_py.py.stamp
        build/catkin_tools_prebuild/catkin_generated/stamps/catkin_tools_prebuild/package.xml.stamp
        build/catkin_tools_prebuild/catkin_generated/stamps/catkin_tools_prebuild/pkg.pc.em.stamp
        build/catkin_tools_prebuild/catkin_generated/env_cached.sh
        build/catkin_tools_prebuild/catkin_generated/generate_cached_setup.py
        build/catkin_tools_prebuild/catkin_generated/package.cmake
        build/catkin_tools_prebuild/catkin_generated/pkg.develspace.context.pc.py
        build/catkin_tools_prebuild/catkin_generated/pkg.installspace.context.pc.py
        build/catkin_tools_prebuild/catkin_generated/setup_cached.sh
        build/catkin_tools_prebuild/gtest/cmake_install.cmake
        build/catkin_tools_prebuild/gtest/CTestTestfile.cmake
        build/catkin_tools_prebuild/gtest/Makefile
        build/catkin_tools_prebuild/cmake_install.cmake
        build/catkin_tools_prebuild/CMakeCache.txt
        build/catkin_tools_prebuild/CMakeLists.txt
        build/catkin_tools_prebuild/CTestTestfile.cmake
        build/catkin_tools_prebuild/Makefile
        build/catkin_tools_prebuild/package.xml
        devel/etc/catkin/profile.d/06-ctr-nuke.sh
        devel/lib/pkgconfig/catkin_tools_prebuild.pc
        devel/share/catkin_tools_prebuild/cmake/catkin_tools_prebuildConfig-version.cmake
        devel/share/catkin_tools_prebuild/cmake/catkin_tools_prebuildConfig.cmake
        devel/_setup_util.py
        devel/env.sh
        devel/setup.bash
        devel/setup.sh
        devel/setup.zsh
        logs/catkin_tools_prebuild/build.cache-manifest.000.log
        logs/catkin_tools_prebuild/build.cache-manifest.log
        logs/catkin_tools_prebuild/build.cmake.000.log
        logs/catkin_tools_prebuild/build.cmake.log
        logs/catkin_tools_prebuild/build.ctr-nuke.000.log
        logs/catkin_tools_prebuild/build.ctr-nuke.log
        logs/catkin_tools_prebuild/build.loadenv.000.log
        logs/catkin_tools_prebuild/build.loadenv.log
        logs/catkin_tools_prebuild/build.make.000.log
        logs/catkin_tools_prebuild/build.make.log
        logs/catkin_tools_prebuild/build.mkdir.000.log
        logs/catkin_tools_prebuild/build.mkdir.001.log
        logs/catkin_tools_prebuild/build.mkdir.log
        logs/catkin_tools_prebuild/build.symlink.000.log
        logs/catkin_tools_prebuild/build.symlink.log
        README.md)

add_executable(SMR1 ${SOURCE_FILES})