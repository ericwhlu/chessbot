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


def test_a_turn_that_is_not_a_quarter_is_not_made_even_when_it_would_be_roomier():
    # only two directions on offer, 45 degrees apart. At the pick the diagonal is tight (2.5 mm),
    # at the place the rank is blocked: the roomiest pair would be (0, 45), a turn of 45 degrees
    pick, place = xy("d4"), xy("g6")
    step = 0.055 / math.sqrt(2.0)
    pick_others = [(pick[0] + step, pick[1] + step)]
    place_others = [xy("f6"), xy("h6")]
    a, b, clear = choose(pick, pick_others, place, place_others, directions=(0.0, 45.0))
    assert (a, b) == (45.0, 45.0)
    assert clear == pytest.approx(0.055 - 0.040 - 0.0125)
    # the forbidden pair really is the roomier one
    forbidden = min(clearance(0.0, pick, pick_others), clearance(45.0, place, place_others))
    assert forbidden > clear + 0.005


def test_among_equally_tight_choices_the_roomier_other_end_wins():
    # 40 mm squares, e7 to e5 with a piece on d4: both diagonals are equally tight at the pick,
    # but at the place one of them points at d4
    def at(name):
        return ((ord(name[0]) - ord("a") + 0.5) * 0.04, (int(name[1]) - 0.5) * 0.04)

    start = [f + r for r in "1278" for f in "abcdefgh"]
    obstacles = [at(n) for n in start if n != "e7"] + [at("d4")]
    a, b, _ = choose(at("e7"), obstacles, at("e5"), obstacles)
    assert (a, b) == (135.0, 135.0)
    assert clearance(135.0, at("e5"), obstacles) > clearance(45.0, at("e5"), obstacles) + 0.02


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


def on_line(angle, degrees):
    """Is ``angle`` (rad) the line at ``degrees``? A line has no sign: 180 is 0."""
    return abs(math.sin(angle - math.radians(degrees))) < 1e-9


@pytest.mark.parametrize("board_yaw, world", [(0.0, 90.0), (30.0, 120.0), (90.0, 0.0), (-60.0, 30.0)])
def test_the_board_yaw_is_added_to_the_direction(board_yaw, world):
    # rook d4 beside a pawn on e4, going to d8 beside the king on e8: at both ends the fingers must
    # open along the file, board direction 90. In the world that is 90 degrees + the board's yaw.
    seen, executor = recorder()
    locations = Locations(frame=BoardFrame(origin_x=0.6, origin_y=-0.2, yaw=math.radians(board_yaw)))
    payload = command("4k3/8/8/8/3RP3/8/8/4K3 w - - 0 1", "d4d8")
    assert run_command(payload, locations, executor)["status"] == "ok"
    assert on_line(seen[0]["pick_grip"], world) and on_line(seen[0]["place_grip"], world)
    assert seen[0]["grip_clearance"] == pytest.approx(plan_grips(payload, Locations())[0][2])   # yaw changes no distance


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
    assert on_line(grips[0][1], 0.0)

    # slots laid out in a row along board -X instead: with nothing in them the fingers open along
    # the row (the default); with the third slot filled they must open across it
    row = Locations(discard_per_column=1)
    assert on_line(plan_grips(command(fen, "e4d5"), row)[0][1], 0.0)
    row.discarded = 3
    grips = plan_grips(command(fen, "e4d5"), row)
    assert on_line(grips[0][0], 90.0) and on_line(grips[0][1], 90.0)     # no turn on the way
    # the second action puts the pawn on d5, far from the discard row: back to the default
    assert on_line(grips[1][0], 0.0) and on_line(grips[1][1], 0.0)


def test_a_discard_slot_filled_by_one_action_is_in_the_way_of_the_next():
    # a capture with promotion discards two pieces; with the slots in a row along board -X the
    # second one is put down right beside the first, so the fingers must open across the row
    payload = command("3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7d8q", allow_promotion=True)
    assert [a["place"] for a in payload["actions"]] == ["DISCARD", "DISCARD", "d8"]
    grips = plan_grips(payload, Locations(discard_per_column=1))
    # the pawn on e7 has the king above it on the file: gripped along the rank, then turned by a
    # quarter on the way, because the rook now stands in the slot beside the one it goes to
    assert on_line(grips[1][0], 0.0) and on_line(grips[1][1], 90.0)


def test_the_pick_and_the_place_each_get_their_own_direction():
    # knight d4 between pawns on c4 and e4 (open along the file), going to f5 between f4 and f6
    # (open along the rank): a quarter turn on the way
    grips = plan_grips(command("4k3/8/5p2/8/2PNPP2/8/8/4K3 w - - 0 1", "d4f5"), Locations())
    assert on_line(grips[0][0], 90.0) and on_line(grips[0][1], 0.0)


def test_a_piece_put_down_by_one_action_is_in_the_way_of_the_next():
    # long castling: the king lands on c1; the rook then goes to d1, right beside it on the rank
    grips = plan_grips(command("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"), Locations())
    assert on_line(grips[0][0], 0.0) and on_line(grips[0][1], 0.0)       # c1: a1 is two squares away
    assert on_line(grips[1][0], 90.0) and on_line(grips[1][1], 90.0)     # d1: the king on c1 is next door
    # short castling: h1 is beside g1 for the king, then g1 is beside both h1 and f1 for the rook
    grips = plan_grips(command("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"), Locations())
    assert all(on_line(a, 90.0) and on_line(b, 90.0) for a, b, _ in grips)


def test_a_piece_taken_away_by_one_action_is_out_of_the_way_of_the_next():
    # en passant: the pawn on d5 stands beside e5 until it is carried off
    payload = command("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6")
    assert payload["actions"] == [{"pick": "d5", "place": "DISCARD"}, {"pick": "e5", "place": "d6"}]
    grips = plan_grips(payload, Locations())
    assert on_line(grips[0][0], 90.0)                                    # d5, with e5 beside it on the rank
    assert on_line(grips[1][0], 0.0) and on_line(grips[1][1], 0.0)       # e5 alone now
    assert grips[1][2] > 0.05


def test_a_promotion_gets_a_direction_for_the_pawn_and_for_the_spare_queen():
    payload = command("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8q", allow_promotion=True)
    assert payload["actions"] == [{"pick": "b7", "place": "DISCARD"}, {"pick": "SPARE_Q_W", "place": "b8"}]
    grips = plan_grips(payload, Locations())
    assert len(grips) == 2
    # nothing stands near b7, the discard slot, the spare or b8: the default, with plenty of room
    assert all(on_line(a, 0.0) and on_line(b, 0.0) and clear > 0.05 for a, b, clear in grips)


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
    # "reach" = joint angles exist for one of the two yaws. Whether the arm gets there in a
    # straight line from where it is, is another question (rank_yaws below).
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


@pytest.mark.parametrize(
    "bx, by, grip, joints",
    [
        (0.236, 0.161, 0.0, [-0.1568, -0.5141, 1.6212, -1.5708, -1.0064, -1.7276]),
        (0.206, -0.029, 45.0, [0.5408, -0.5141, 1.6211, -1.5708, -1.0064, 2.1116]),
        (0.395, -0.065, 135.0, [-0.1568, -0.5141, 1.6212, -1.5708, -1.0064, -1.7276]),
    ],
)
def test_a_yaw_that_ends_next_to_the_wrist_limit_is_the_second_choice(bx, by, grip, joints):
    line = math.radians(grip)
    ends = {}
    for yaw in K.grip_yaws(line):
        near = K.nearest_solution((bx, by, TRAVEL_Z), K.top_down_rotation(yaw), joints)
        assert near is not None                          # the arm can turn to both yaws from here
        ends[yaw] = near
    room = {yaw: K.JOINT_LIMITS[5] - abs(found[5]) for yaw, (_, found) in ends.items()}
    nearer = min(ends, key=lambda yaw: ends[yaw][0])
    assert room[nearer] < K.WRIST_MARGIN                 # the smaller motion ends at the limit ...
    first = K.rank_yaws(bx, by, TRAVEL_Z, line, joints, GRIP_Z)[0]
    assert first != nearer and room[first] > K.WRIST_MARGIN   # ... so the other yaw goes first


def test_the_yaw_the_gripper_has_is_kept_even_next_to_the_wrist_limit():
    # seen with MoveIt on mock hardware, e2 -> e4 with the board along the arm's x axis: keeping the
    # yaw ends 7 degrees from the limit. The other yaw is half a turn away, and the straight line
    # that was tried first for it stopped after 11 % (MoveIt turned the tool towards the limit).
    yaw = math.radians(-45.0)
    line = math.radians(45.0)
    joints = K.nearest_solution((0.397, -0.125, 0.251), K.top_down_rotation(yaw), K.WORK)[1]
    kept = K.nearest_solution((0.397, -0.025, 0.251), K.top_down_rotation(yaw), joints)
    assert K.JOINT_LIMITS[5] - abs(kept[1][5]) < K.WRIST_MARGIN
    assert K.rank_yaws(0.397, -0.025, 0.251, line, joints, 0.124, yaw)[0] == pytest.approx(yaw)
    assert K.yaw_candidates(0.397, -0.025, 0.251, line, joints, 0.124, yaw, straight=True)[0] == pytest.approx(yaw)
    # when the gripper's yaw is not known (after a joint-space move) the roomier yaw goes first
    assert K.rank_yaws(0.397, -0.025, 0.251, line, joints, 0.124)[0] != pytest.approx(yaw)


def test_the_wide_search_is_skipped_when_the_arm_can_turn_to_one_of_the_yaws(monkeypatch):
    line = math.radians(90.0)
    yaw = K.grip_yaws(line)[0]
    above = K.top_down_solutions(0.35, 0.10, TRAVEL_Z, yaw)[0]
    monkeypatch.setattr(K, "top_down_solutions", lambda *a, **k: pytest.fail("searched from many starting guesses"))
    ranked = K.rank_yaws(0.30, -0.05, TRAVEL_Z, line, above, GRIP_Z, yaw)
    assert sorted(ranked) == sorted(K.grip_yaws(line))


# ------------------------------------------------------------- the yaws tried for one pose move
def test_without_a_grip_direction_the_lab_orientation_is_the_only_candidate():
    assert K.yaw_candidates(0.35, 0.10, GRIP_Z, None) == [0.0]
    assert K.yaw_candidates(0.35, 0.10, GRIP_Z, None, None, None, 1.0, straight=True) == [0.0]


def test_a_descent_or_a_lift_never_turns_the_gripper():
    for degrees in (0.0, 45.0, 90.0, 135.0):
        line = math.radians(degrees)
        first, second = K.grip_yaws(line)
        for yaw, other in ((first, second), (second, first)):
            # joint angles that belong to the OTHER yaw must not talk it into turning
            for joints in (None, K.top_down_solutions(0.35, 0.10, TRAVEL_Z, other)[0]):
                got = K.yaw_candidates(0.35, 0.10, GRIP_Z, line, joints, None, yaw, straight=True)
                assert got == [pytest.approx(yaw)]


def test_a_carry_may_try_both_yaws():
    line = math.radians(90.0)
    yaw = K.grip_yaws(line)[0]
    above = K.top_down_solutions(0.35, 0.10, TRAVEL_Z, yaw)[0]
    both = sorted(K.grip_yaws(line))
    # a straight line to above another square (the descent comes next): both, the present yaw first
    got = K.yaw_candidates(0.30, -0.05, TRAVEL_Z, line, above, GRIP_Z, yaw, straight=True)
    assert sorted(got) == both and got[0] == pytest.approx(yaw)
    # a free-space move: both
    assert sorted(K.yaw_candidates(0.30, -0.05, TRAVEL_Z, line, above, None, yaw, straight=False)) == both
    # a straight move from a yaw that is not on the line (a quarter turn was asked for): both
    quarter = line + math.pi / 2
    assert sorted(K.yaw_candidates(0.35, 0.10, TRAVEL_Z, quarter, above, None, yaw, straight=True)) == sorted(
        K.grip_yaws(quarter)
    )
    # after a joint-space move the tool's yaw is not known: both
    assert sorted(K.yaw_candidates(0.35, 0.10, GRIP_Z, line, above, None, None, straight=True)) == both

