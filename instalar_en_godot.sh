#!/bin/bash
git clone https://github.com/godotengine/godot.git --depth 1
cp -r vcpi_bridge godot/modules/
cd godot
scons platform=linuxbsd target=editor
