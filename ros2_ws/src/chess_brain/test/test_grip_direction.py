"""The direction the gripper opens along: the chooser, the occupied squares in the command, what
arm_adapter hands to the executor, and the tool yaw in the Gen3 Lite kinematics."""

import math

import chess
import pytest

from chess_brain import gen3_lite_kinematics as K
from chess_brain.actions import actions_payload, move_to_actions, occupied_squares
from chess_brain.arm_adapter import plan_grips, run_command
from chess_brain.board_frame import BoardFrame, Locations
from chess_brain.grip_direction import DIRECTIONS, Fingers, choose, clearance

P = 0.05
START = [f + r for r in "1278" for f in "abcdefgh"]


def xy(name):
    return ((ord(name[0]) - ord("a") + 0.5) * P, (int(name[1]) - 0.5) * P)


def others(squares, *without):
    return [xy(name) for name in squares if name not in without]


# ------------------------------------------------------------------------------- the model
def test_clearance_is_infinite_with_nothing_around():
    assert clearance(0.0, xy("d4"), []) == math.inf


def test_clearance_along_the_opening_line_matches_the_geometry():
    f = Fingers()
    # a neighbour one square away on the line: finger's outer face at gap/2 + thickness
    expected = P - (f.gap / 2 + f.thickness) - f.piece_radius
    assert clearance(0.0, xy("d4"), [xy("e4")], f) == pytest.approx(expected)
    assert clearance(0.0, xy("d4"), [xy("c4")], f) == pytest.approx(expected)      # the fingers are alike
    assert clearance(180.0, xy("d4"), [xy("e4")], f) == pytest.approx(expected)    # a line has no sign


def test_clearance_across_the_opening_line_matches_the_geometry():
    f = Fingers()
    # a neighbour one square away across the line passes beside the fingers
    da = f.gap / 2            # its centre is this far short of the finger along the line ...
    dc = P - f.width / 2      # ... and this far to the side
    assert clearance(90.0, xy("d4"), [xy("e4")], f) == pytest.approx(math.hypot(da, dc) - f.piece_radius)


def test_a_piece_under_a_finger_counts_as_overlap():
    assert clearance(0.0, (0.0, 0.0), [(0.03, 0.0)]) < -Fingers().piece_radius


def test_the_model_ranks_a_crowded_square_like_the_3d_study():
    # all eight neighbours present: the 3D study found about 4 to 6 mm along the grid and 8 to 9 mm
    # on the diagonal; the plan view must agree on which is better, and by a similar margin
    ring = [xy(n) for n in ("c3", "c4", "c5", "d3", "d5", "e3", "e4", "e5")]
    for gap in (0.030, 0.040, 0.056):
        f = Fingers(gap=gap)
        along = max(clearance(d, xy("d4"), ring, f) for d in (0.0, 90.0))
        diagonal = min(clearance(d, xy("d4"), ring, f) for d in (45.0, 135.0))
        assert diagonal - along > 0.005
        assert 0.008 < diagonal < 0.015


@pytest.mark.parametrize("field", ["thickness", "width", "gap", "piece_radius", "enough"])
def test_fingers_must_have_positive_sizes(field):
    with pytest.raises(ValueError):
        Fingers(**{field: 0.0})


# ------------------------------------------------------------------------------- the choice
def test_alone_on_the_board_the_gripper_stays_along_the_grid_and_does_not_turn():
    assert choose(xy("d4"), [], xy("d5"), [])[:2] == (0.0, 0.0)


def test_a_pawn_in_the_starting_row_is_gripped_on_the_diagonal():
    pick, place, clear = choose(xy("e7"), others(START, "e7"), xy("e5"), others(START, "e7"))
    assert pick in (45.0, 135.0) and place == pick
    assert clear > 0.008


def test_with_room_on_one_side_the_fingers_open_that_way():
    # a neighbour on the same rank: open along the file, and the other way round
    assert choose(xy("d4"), [xy("e4")], xy("d8"), [xy("e4")])[:2] == (90.0, 90.0)
    assert choose(xy("d4"), [xy("d5")], xy("h4"), [xy("d5")])[:2] == (0.0, 0.0)


def test_enough_is_enough():
    # far-away pieces do not make the gripper turn: both directions have more than `enough`
    far = [xy("a1"), xy("h8")]
    assert choose(xy("d4"), far, xy("d5"), far)[:2] == (0.0, 0.0)


def test_the_piece_is_turned_by_a_quarter_or_not_at_all():
    squares = [f + r for f in "abcdefgh" for r in "12345678"]
    import random

    rng = random.Random(4)
    for _ in range(300):
        occupied = rng.sample(squares, rng.randint(2, 30))
        pick = occupied[0]
        free = [s for s in squares if s not in occupied]
        place = rng.choice(free)
        obstacles = others(occupied, pick)
        a, b, clear = choose(xy(pick), obstacles, xy(place), obstacles)
        assert a in DIRECTIONS and b in DIRECTIONS
        turn = abs(a - b) % 180.0
        assert min(turn, 180.0 - turn) in (0.0, 90.0)
        # never worse than the best single direction that needs no turn
        no_turn = max(min(clearance(d, xy(pick), obstacles), clearance(d, xy(place), obstacles)) for d in DIRECTIONS)
        assert min(clear, Fingers().enough) >= min(no_turn, Fingers().enough) - 1e-9


def test_a_quarter_turn_is_used_when_it_helps():
    # at the pick the rank is crowded, at the place the file is: open along the file, then the rank
    pick_others = [xy("c4"), xy("e4")]
    place_others = [xy("g7"), xy("g5")]
    a, b, _ = choose(xy("d4"), pick_others + place_others, xy("g6"), pick_others + place_others)
    assert (a, b) == (90.0, 0.0)


def test_the_choice_is_the_same_every_time():
    obstacles = others(START, "g8")
    assert len({choose(xy("g8"), obstacles, xy("f6"), obstacles) for _ in range(5)}) == 1


# ------------------------------------------------------------------- the command from the brain
def test_occupied_squares_lists_every_piece():
    board = chess.Board()
    assert sorted(occupied_squares(board)) == sorted(START)
    board.push_san("e4")
    assert "e4" in occupied_squares(board) and "e2" not in occupied_squares(board)


def test_the_payload_carries_the_occupied_squares_only_when_given():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    actions = move_to_actions(board, move)
    assert "occupied" not in actions_payload(1, move, actions)
    payload = actions_payload(1, move, actions, occupied_squares(board))
    assert payload["occupied"] == sorted(START)
    assert payload["actions"] == [{"pick": "e2", "place": "e4"}]   # the actions themselves are unchanged


# ------------------------------------------------------------------- what the executor is told
def command(fen, uci, action_id=1, allow_promotion=False):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    return actions_payload(action_id, move, move_to_actions(board, move, allow_promotion), occupied_squares(board))


def recorder():
    seen = []

    def executor(pick_xyz, place_xyz, info):
        seen.append(dict(info))

    return seen, executor


def test_no_occupied_list_no_grip_direction():
    seen, executor = recorder()
    payload = {"id": 1, "move": "e7e5", "actions": [{"pick": "e7", "place": "e5"}]}
    assert plan_grips(payload, Locations()) is None
    assert run_command(payload, Locations(), executor)["status"] == "ok"
    assert "pick_grip" not in seen[0] and "place_grip" not in seen[0]


def test_grips_can_be_switched_off():
    seen, executor = recorder()
    run_command(command(chess.STARTING_FEN, "e2e4"), Locations(), executor, grips=False)
    assert "pick_grip" not in seen[0]


def test_the_executor_gets_world_frame_angles():
    # board turned by 90 degrees: a direction along the board's diagonal is 45 + 90 in the world
    seen, executor = recorder()
    locations = Locations(frame=BoardFrame(origin_x=0.6, origin_y=-0.2, yaw=math.radians(90.0)))
    assert run_command(command(chess.STARTING_FEN, "e2e4"), locations, executor)["status"] == "ok"
    info = seen[0]
    board_direction = math.degrees(info["pick_grip"] - locations.frame.yaw) % 180.0
    assert board_direction in (45.0, 135.0)              # a pawn in the starting row: diagonal
    assert info["place_grip"] == info["pick_grip"]       # no reason to turn on the way to e4
    assert info["grip_clearance"] > 0.008


def test_a_capture_takes_the_captured_piece_out_of_the_way_first():
    # white pawn e4 takes d5; every square around d5 and e4 is otherwise empty
    fen = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"
    payload = command(fen, "e4d5")
    assert payload["actions"] == [{"pick": "d5", "place": "DISCARD"}, {"pick": "e4", "place": "d5"}]
    grips = plan_grips(payload, Locations())
    # action 1: the pawn on d5 is gripped with the e4 pawn next to it on the diagonal: not along it
    first_pick = math.degrees(grips[0][0]) % 180.0
    assert first_pick != 135.0
    # action 2: d5 is empty now and nothing is near e4 or d5: along the grid, no turn
    assert (math.degrees(grips[1][0]) % 180.0, math.degrees(grips[1][1]) % 180.0) == (0.0, 0.0)
    assert grips[1][2] > 0.10          # only the two kings are left, far away


def test_filled_discard_slots_are_in_the_way_too():
    locations = Locations()
    locations.discarded = 3            # three captured pieces stand in the discard column already
    fen = "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1"
    grips = plan_grips(command(fen, "e4d5"), locations)
    # the fourth slot has neighbours along the column (board +Y): put the piece down opening across it
    assert math.degrees(grips[0][1]) % 180.0 == 0.0


def test_en_passant_castling_and_promotion_get_a_direction_for_every_action():
    cases = [
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", False, 2),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", False, 2),
        ("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8q", True, 2),
    ]
    for fen, uci, promotion, count in cases:
        payload = command(fen, uci, allow_promotion=promotion)
        grips = plan_grips(payload, Locations())
        assert len(grips) == len(payload["actions"]) == count
        assert all(math.isfinite(a) and math.isfinite(b) for a, b, _ in grips)


@pytest.mark.parametrize("bad", ["e2", {"e2": 1}, ["e2", "z9"], [3]])
def test_a_malformed_occupied_list_is_a_hint_that_cannot_be_used(bad):
    payload = {"id": 4, "move": "e7e5", "actions": [{"pick": "e7", "place": "e5"}], "occupied": bad}
    with pytest.raises(ValueError, match="occupied"):
        plan_grips(payload, Locations())
    # ... but it does not stop the game: the move is made with the arm's own orientation
    seen, executor = recorder()
    assert run_command(payload, Locations(), executor)["status"] == "ok"
    assert len(seen) == 1 and "pick_grip" not in seen[0]


def test_off_board_locations_have_board_frame_positions():
    locations = Locations()
    assert locations.board_xy("e4") == pytest.approx((0.225, 0.175))
    assert locations.board_xy("DISCARD") == pytest.approx(locations.discard_first)
    assert locations.board_xy("DISCARD", discard_offset=2) == pytest.approx((-0.075, 0.125))
    assert locations.board_xy("SPARE_Q_W") == pytest.approx(locations.spares["SPARE_Q_W"])


# ------------------------------------------------------------------- tool yaw in the kinematics
def quaternion_matrix(q):
    x, y, z, w = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def test_yaw_zero_is_the_lab_orientation():
    assert K.top_down_quaternion(0.0) == pytest.approx(K.TOP_DOWN)
    assert K.top_down_rotation(0.0) == K.TOP_DOWN_ROTATION


@pytest.mark.parametrize("degrees", [-170, -90, -30, 0, 45, 90, 135, 180])
def test_the_quaternion_and_the_matrix_are_the_same_rotation(degrees):
    yaw = math.radians(degrees)
    q, r = quaternion_matrix(K.top_down_quaternion(yaw)), K.top_down_rotation(yaw)
    for i in range(3):
        assert q[i] == pytest.approx(r[i], abs=1e-12)
    assert [r[i][2] for i in range(3)] == [0.0, 0.0, -1.0]         # still pointing straight down
    # the fingers open along the tool's y axis: at angle yaw + 90 degrees in the base frame
    line = math.atan2(r[1][1], r[0][1])
    assert math.sin(line - (yaw + math.pi / 2)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("degrees", [0, 30, 45, 90, 135, 200])
def test_both_yaws_of_a_grip_line_open_along_it(degrees):
    grip = math.radians(degrees)
    first, second = K.grip_yaws(grip)
    assert abs(math.cos(first - second) + 1.0) < 1e-12            # half a turn apart
    for yaw in (first, second):
        r = K.top_down_rotation(yaw)
        assert abs(math.sin(math.atan2(r[1][1], r[0][1]) - grip)) < 1e-12


def test_solutions_with_a_yaw_reach_the_turned_pose():
    x, y, z, yaw = 0.35, 0.10, 0.121, math.radians(45.0)
    found = K.top_down_solutions(x, y, z, yaw)
    assert found
    for joints in found:
        distance, angle = K.pose_error(joints, (x, y, z), K.top_down_rotation(yaw))
        assert distance < K.POSITION_TOLERANCE and angle < K.ROTATION_TOLERANCE
    # and they are not the solutions of the unturned pose
    assert K.pose_error(found[0], (x, y, z), K.TOP_DOWN_ROTATION)[1] > 0.5


@pytest.mark.parametrize(
    "x, y, z, reachable",
    [
        (0.447, 0.175, 0.251, True),
        (0.197, -0.175, 0.121, True),
        (0.497, 0.175, 0.121, True),     # near the wrist's limit: only one yaw of each pair works
        (0.547, 0.025, 0.121, False),    # the far rank of the 400 mm board
    ],
)
def test_reach_does_not_depend_on_the_grip_direction(x, y, z, reachable):
    for direction in (0.0, 45.0, 90.0, 135.0):
        either = any(K.top_down_solutions(x, y, z, yaw) for yaw in K.grip_yaws(math.radians(direction)))
        assert either is reachable


# ------------------------------------------------------------------- which of the two yaws
TRAVEL_Z = -0.056 + 0.157 + 0.150
GRIP_Z = -0.056 + 0.157 + 0.020


def _feasible(x, y, z, yaw, joints):
    return K.nearest_solution((x, y, z), K.top_down_rotation(yaw), joints) is not None


def test_going_straight_down_keeps_the_yaw_the_arm_has():
    for degrees in (0.0, 45.0, 90.0, 135.0):
        grip = math.radians(degrees)
        for yaw in K.grip_yaws(grip):
            above = K.top_down_solutions(0.35, 0.10, TRAVEL_Z, yaw)
            assert above
            ranked = K.rank_yaws(0.35, 0.10, GRIP_Z, grip, above[0])
            assert ranked[0] == pytest.approx(yaw)


def test_the_first_yaw_is_one_the_arm_can_turn_to_from_where_it_is():
    """Moves between squares for boards turned different ways: whenever one of the two yaws can
    be reached without leaving the arm's configuration, that one is ranked first. The smaller
    turn of the tool is not always that one (the last joint has limits)."""
    import random

    rng = random.Random(11)
    checked = differs_from_smaller_turn = 0
    for _ in range(40):
        board_yaw = rng.choice([0.0, math.pi / 2, -math.pi / 2, math.pi])
        ax, ay = rng.uniform(0.20, 0.47), rng.uniform(-0.17, 0.17)
        bx, by = rng.uniform(0.20, 0.47), rng.uniform(-0.17, 0.17)
        grip_a = board_yaw + math.radians(rng.choice([0.0, 45.0, 90.0, 135.0]))
        grip_b = grip_a + rng.choice([0.0, math.pi / 2])
        yaw_a = K.grip_yaws(grip_a)[rng.randrange(2)]
        start = K.top_down_solutions(ax, ay, TRAVEL_Z, yaw_a)
        if not start:
            continue
        joints = start[rng.randrange(len(start))]
        ranked = K.rank_yaws(bx, by, TRAVEL_Z, grip_b, joints, GRIP_Z, yaw_a)
        assert sorted(ranked) == sorted(K.grip_yaws(grip_b))
        can = [_feasible(bx, by, TRAVEL_Z, yaw, joints) for yaw in ranked]
        if any(can):
            checked += 1
            assert can[0], "a yaw the arm cannot turn to was ranked first although the other one is fine"
            smaller = min(ranked, key=lambda yaw: abs(math.atan2(math.sin(yaw - yaw_a), math.cos(yaw - yaw_a))))
            differs_from_smaller_turn += ranked[0] != smaller
    assert checked > 20
    assert differs_from_smaller_turn > 0     # the rule matters: "the smaller turn" would be wrong here


def test_without_joint_angles_the_yaw_nearer_to_the_last_one_comes_first():
    grip = math.radians(90.0)                      # yaws 0 and 180 degrees
    assert K.rank_yaws(0.35, 0.0, TRAVEL_Z, grip, None, None, 0.1)[0] == pytest.approx(0.0)
    assert abs(K.rank_yaws(0.35, 0.0, TRAVEL_Z, grip, None, None, 3.0)[0]) == pytest.approx(math.pi)


def test_a_yaw_with_no_solution_goes_last():
    # near the wrist's limit only one yaw of the pair has joint angles (see the reach test above)
    x, y, z = 0.497, 0.175, 0.121
    for degrees in (0.0, 45.0, 90.0, 135.0):
        grip = math.radians(degrees)
        has = {yaw: bool(K.top_down_solutions(x, y, z, yaw)) for yaw in K.grip_yaws(grip)}
        if sum(has.values()) == 1:
            assert has[K.rank_yaws(x, y, z, grip)[0]]
            return
    pytest.fail("expected a direction for which only one yaw is reachable at this point")
