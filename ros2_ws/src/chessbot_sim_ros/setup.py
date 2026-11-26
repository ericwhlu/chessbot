from glob import glob

from setuptools import find_packages, setup

package_name = "chessbot_sim_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="David Zeng",
    maintainer_email="todo@todo.todo",
    description="chess_brain playing in the MuJoCo simulation (chessbot_sim) over ROS 2.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mujoco_world_node = chessbot_sim_ros.mujoco_world_node:main",
            "dry_run = chessbot_sim_ros.dry_run:main",
        ],
    },
)
