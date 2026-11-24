"""Build the Chessbot board, Gen3, and calibrated target geometry in MuJoCo."""
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BOARD_SDF = REPO_ROOT / "chessboard.sdf"
ASSETS = Path(__file__).resolve().with_name("assets")
WIDTH, HEIGHT = 1120, 840


def add_sdf_board(root):
    """Import the static board link's mesh, collision box, and square visuals."""
    source = ET.parse(BOARD_SDF)
    link = source.find("./world/model[@name='chessboard']/link[@name='board_link']")
    collision = link.find("collision[@name='board_collision']")
    base = link.find("visual[@name='board_visual']")
    mesh = base.find("geometry/mesh")
    mesh_uri = mesh.findtext("uri").removeprefix("model://")
    mesh_path = REPO_ROOT / "models" / mesh_uri
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="board_mesh", file=mesh_path.name, scale=mesh.findtext("scale"))
    board = ET.SubElement(root.find("worldbody"), "body", name="chessboard")
    ET.SubElement(board, "geom", name="board_visual", type="mesh", mesh="board_mesh",
                  rgba=base.findtext("material/diffuse"), contype="0", conaffinity="0")

    def add_box(element, **attrs):
        # Board-link poses in this SDF contain XYZ and zero rotation.
        xyz = np.fromstring(element.findtext("pose"), sep=" ")[:3]
        half_size = np.fromstring(element.findtext("geometry/box/size"), sep=" ") / 2
        ET.SubElement(board, "geom", name=element.attrib["name"], type="box",
                      pos=xml_values(xyz), size=xml_values(half_size), **attrs)
        return xyz, half_size

    center, half_size = add_box(collision, rgba="0 0 0 0")
    squares = {}
    for visual in link.findall("visual"):
        if visual.attrib["name"].startswith("square_"):
            xyz, half_size = add_box(visual, rgba=visual.findtext("material/diffuse"),
                                     contype="0", conaffinity="0")
            xyz[2] += half_size[2]
            squares[visual.attrib["name"].removeprefix("square_")] = xyz
    square_tops = [xyz[2] for xyz in squares.values()]
    if len(square_tops) != 64 or not np.allclose(square_tops, square_tops[0]):
        raise ValueError("Expected 64 coplanar square visuals in the Chessbot SDF")
    center[2] = square_tops[0]
    return board, center, half_size, squares, {mesh_path.name: mesh_path.read_bytes()}


def xml_values(values):
    return " ".join(str(value) for value in values)


BACK_RANK = ("rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook")
STARTING_RANKS = {1: BACK_RANK, 2: ("pawn",) * 8, 7: ("pawn",) * 8, 8: BACK_RANK}


def add_chess_pieces(root, board, squares, assets, movable):
    asset = root.find("asset")
    ET.SubElement(asset, "material", name="white_piece", rgba="0.96 0.83 0.64 1", specular="0.30")
    ET.SubElement(asset, "material", name="black_piece", rgba="0.21 0.27 0.36 1", specular="0.30")
    for name in sorted(set(BACK_RANK) | {"pawn"}):
        filename = f"{name}.stl"
        assets[filename] = (ASSETS / "pieces" / filename).read_bytes()
        ET.SubElement(asset, "mesh", name=name, file=filename, scale=".001 .001 .001")
    initial_free_positions = []
    for rank, names in STARTING_RANKS.items():
        white = rank <= 2
        material = "white_piece" if white else "black_piece"
        for file, name in zip("abcdefgh", names):
            square = f"{file}{rank}"
            attributes = {"name": f"piece_{square}", "type": "mesh", "mesh": name,
                          "euler": f"0 0 {0 if white else np.pi}", "material": material}
            if movable:
                obj = ET.SubElement(root.find("worldbody"), "body", name=f"object_{square}",
                                    pos=xml_values(squares[square]))
                ET.SubElement(obj, "freejoint", name=f"free_{square}")
                ET.SubElement(obj, "geom", **attributes, mass="0.032", friction="1 0.02 0.001",
                              contype="2", conaffinity="3")
                initial_free_positions.extend([*squares[square], 1, 0, 0, 0])
            else:
                ET.SubElement(board, "geom", **attributes, pos=xml_values(squares[square]),
                              contype="0", conaffinity="0")
    if movable:
        for key in root.findall("keyframe/key"):
            key.set("qpos", f"{key.attrib['qpos']} {xml_values(initial_free_positions)}")


def add_board_tag_mounts(board, tags, center, half_size):
    for tag in tags:
        x, y = tag["center_xy_m"]
        edge = np.clip(y, center[1] - half_size[1], center[1] + half_size[1])
        if y == edge:
            continue
        ET.SubElement(board, "geom", name=f"tag_mount_{tag['id']}", type="box",
                      pos=xml_values([x, (edge + y) / 2, half_size[2]]),
                      size=xml_values([0.025, abs(y - edge) / 2, half_size[2]]),
                      rgba="0.60 0.43 0.27 1")


def add_arm(root, config, assets):
    directory = ASSETS / "kinova_gen3"
    source = ET.parse(directory / "gen3.xml").getroot()
    for section in ("option", "default", "actuator", "keyframe"):
        root.append(source.find(section))
    for mesh in source.findall("asset/mesh"):
        filename = mesh.attrib["file"]
        assets[filename] = (directory / "assets" / filename).read_bytes()
        root.find("asset").append(mesh)

    base = source.find("worldbody/body")
    placement = config["base_in_board"]
    position = np.asarray(placement["position_m"], dtype=float)
    heading = np.deg2rad(placement["yaw_deg"])
    base.set("pos", xml_values(position))
    base.set("euler", f"0 0 {heading}")
    root.find("worldbody").append(base)
    add_gripper(root, base.find(".//body[@name='bracelet_link']"), assets)

    z_half = position[2] / 2
    ET.SubElement(base, "geom", name="arm_mount_plate", type="box",
                  size=xml_values([*config["plate_half_size_xy_m"], z_half]),
                  pos=f"0 0 {-z_half}", rgba="0.40 0.45 0.50 1")
    return base, position


def add_gripper(root, wrist, assets):
    directory = ASSETS / "robotiq_2f85"
    source = ET.parse(directory / "2f85.xml").getroot()
    gripper = source.find("worldbody/body/body")
    defaults = source.find("default/default")
    # Both upstream models define visual and collision defaults in the global class namespace.
    for element in (*defaults.iter(), *gripper.iter()):
        name = element.get("class")
        if name in ("visual", "collision"):
            element.set("class", f"gripper_{name}")
    root.find("default").append(defaults)

    for asset in source.find("asset"):
        filename = asset.get("file")
        if filename == "base_mount.stl":
            continue
        if filename is not None:
            assets[filename] = (directory / "assets" / filename).read_bytes()
        root.find("asset").append(asset)

    # The Gen3 wrist already includes the mechanical coupling.
    gripper.set("pos", "0 0 -0.06149039")
    gripper.set("quat", "0 -1 1 0")
    gripper.set("childclass", "2f85")
    wrist.append(gripper)
    wrist.find("site[@name='pinch_site']").set("pos", "0 0 -0.181525")

    root.find("option").attrib.update(source.find("option").attrib)
    for section in ("contact", "tendon", "equality"):
        root.append(source.find(section))
    actuators = list(source.find("actuator"))
    root.find("actuator").extend(actuators)
    for key in root.findall("keyframe/key"):
        key.set("qpos", key.attrib["qpos"] + " 0" * len(gripper.findall(".//joint")))
        key.set("ctrl", key.attrib["ctrl"] + " 0" * len(actuators))


def add_target(parent, tags, dictionary, pad_top, pad_thickness):
    cells = dictionary.markerSize + 2
    ink_thickness = 0.00004
    tag_z = pad_top + ink_thickness
    # The decoder returns the upright marker's corners starting at its top left.
    unit_corners = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]], dtype=float) / 2

    def tile(center, half_side, top, thickness, shade, yaw, **attrs):
        ET.SubElement(parent, "geom", type="box",
                      pos=xml_values([*center, top - thickness / 2]),
                      size=xml_values([half_side, half_side, thickness / 2]),
                      euler=f"0 0 {yaw}", rgba=f"{shade} {shade} {shade} 1", **attrs)

    corners_by_id = {}
    for tag in tags:
        center = np.asarray(tag["center_xy_m"], dtype=float)
        size = tag["size_m"]
        yaw = np.deg2rad(tag.get("rotation_deg", 0.0))
        if center.shape != (2,) or not np.isfinite([*center, size, yaw]).all() or size <= 0:
            raise ValueError("Each tag needs a finite XY center, positive size and finite rotation")
        rotation = cv2.Rodrigues(np.array([0.0, 0.0, yaw]))[0][:2, :2]
        corners = center + (unit_corners * size) @ rotation.T
        corners_by_id[tag["id"]] = np.column_stack((corners, np.full(4, tag_z)))
        cell = size / cells
        tile(center, size / 2 + 2 * cell, pad_top, pad_thickness, 1, yaw,
             name=f"tag_pad_{tag['id']}")
        bitmap = cv2.aruco.generateImageMarker(dictionary, tag["id"], cells)
        for row, col in np.argwhere(bitmap == 0):
            offset = np.array([col + 0.5 - cells / 2, cells / 2 - row - 0.5]) * cell
            tile(center + rotation @ offset, cell / 2, tag_z, ink_thickness, 0, yaw,
                 contype="0", conaffinity="0")
    return corners_by_id


def make_scene(board_config, arm_config, fovy, *, movable_pieces=False):
    dictionary_id = getattr(cv2.aruco, "DICT_APRILTAG_" + board_config["tag_family"], None)
    if dictionary_id is None:
        raise ValueError("Unknown AprilTag family")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board_tags, arm_tags = board_config["tags"], arm_config["tags"]
    ids = [tag["id"] for tag in board_tags + arm_tags]
    if len(board_tags) != 4 or len(arm_tags) != 4 or len(set(ids)) != 8 or any(
        not 0 <= tag_id < len(dictionary.bytesList) for tag_id in ids
    ):
        raise ValueError("Provide four tags per target with eight distinct IDs from the same family")

    root = ET.fromstring(f"""
        <mujoco model="Chessbot board and Kinova Gen3">
          <compiler angle="radian"/>
          <visual>
            <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
            <quality offsamples="4"/>
            <headlight ambient=".78 .78 .78" diffuse=".38 .38 .38" specular=".22 .22 .22"/>
          </visual>
          <worldbody>
            <geom type="plane" size="0 0 0.1" rgba=".25 .29 .33 1"/>
            <body name="camera_rig" mocap="true">
              <camera name="rgb" fovy="{fovy}"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    board, board_center, board_half_size, squares, assets = add_sdf_board(root)
    base, base_position = add_arm(root, arm_config, assets)
    if movable_pieces:
        # Represent the Gen3's gravity-compensated joint servo in the manipulation scene.
        for link in base.iter("body"):
            link.set("gravcomp", "1")
        for i in range(1, 8):
            joint = base.find(f".//joint[@name='joint_{i}']")
            joint.set("actuatorgravcomp", "true")
            joint.set("actuatorfrcrange", "-105 105" if i <= 4 else "-52 52")
        # These adjacent convex hulls overlap at the physical shoulder bearing.
        ET.SubElement(root.find("contact"), "exclude", body1="base_link", body2="shoulder_link")
        for name in ("pad_box1", "pad_box2"):
            pad = root.find(f".//default[@class='{name}']/geom")
            pad.set("condim", "4")
            pad.set("friction", "0.9 0.004 0.0001")
    add_chess_pieces(root, board, squares, assets, movable_pieces)
    board_corners = add_target(board, board_tags, dictionary, board_center[2], board_center[2])
    add_board_tag_mounts(board, board_tags, board_center, board_half_size)
    base_corners = add_target(base, arm_tags, dictionary, pad_top=0.0001, pad_thickness=0.0001)

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    view_center = (board_center + base_position) / 2
    view_center[1] += 0.09
    view_center[2] = 0.15
    return model, detector, board_corners, base_corners, view_center
