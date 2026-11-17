#!/usr/bin/env bash
# Hand-drive one full turn against a running brain_node, playing the parts of
# vision and the arm. Start the node first, in another shell, with:
#
#   ros2 run chess_brain brain_node --ros-args -p robot_color:=black -p first_action_id:=0
#
# (first_action_id:=0 makes the first arm_actions id 1, which this script answers.)
set -e

pub() {  # pub <topic> <string>
  ros2 topic pub --once "$1" std_msgs/msg/String "{data: '$2'}" >/dev/null
}

echo "[demo] 1/4 vision: start position"
pub /chessbot/detected_fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
sleep 2

echo "[demo] 2/4 vision: the human played e2e4"
pub /chessbot/detected_fen 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR'
sleep 3   # stable_time + think_time: the node now logs arm_actions with id 1

echo "[demo] 3/4 arm: done, id 1"
ros2 topic pub --once /chessbot/arm_done std_msgs/msg/String '{data: "{\"id\": 1, \"status\": \"ok\"}"}' >/dev/null
sleep 1

echo "[demo] 4/4 board state and status as the brain sees them now:"
ros2 topic echo --once --full-length /chessbot/board_state std_msgs/msg/String \
  --qos-reliability reliable --qos-durability transient_local
ros2 topic echo --once --full-length /chessbot/status std_msgs/msg/String \
  --qos-reliability reliable --qos-durability transient_local
echo "[demo] expected: black has moved, it is white's turn, state wait_human"
