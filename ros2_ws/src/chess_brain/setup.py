from glob import glob

from setuptools import find_packages, setup

package_name = "chess_brain"

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
    description="Chess rules, game-state tracking and Stockfish move selection for the Chessmate robot.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "brain_node = chess_brain.brain_node:main",
            "sim_world_node = chess_brain.sim_world_node:main",
            "operator_node = chess_brain.operator_node:main",
            "chess_cli = chess_brain.cli:main",
        ],
    },
)
