"""Tessellate the Chessbot STEP pieces for the MuJoCo viewer."""
from pathlib import Path

from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.gp import gp_Trsf, gp_Vec
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.StlAPI import StlAPI_Writer


MESH_DIR = Path(__file__).resolve().parent.parent / "models/chessbot/meshes"
OUTPUT_DIR = Path(__file__).resolve().with_name("assets") / "pieces"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = StlAPI_Writer()
    writer.ASCIIMode = False
    # The STEP pieces share a square-centered CAD origin at (25, 25, 5) mm.
    to_square_center = gp_Trsf()
    to_square_center.SetTranslation(gp_Vec(-25, -25, -5))
    for source in sorted(MESH_DIR.glob("*.step")):
        reader = STEPControl_Reader()
        if reader.ReadFile(str(source)) != IFSelect_RetDone or not reader.TransferRoots():
            raise RuntimeError(f"Could not load {source}")
        shape = BRepBuilderAPI_Transform(reader.OneShape(), to_square_center, True).Shape()
        BRepMesh_IncrementalMesh(shape, 0.10, False, 0.3)
        if not writer.Write(shape, str(OUTPUT_DIR / f"{source.stem}.stl")):
            raise RuntimeError(f"Could not tessellate {source}")


if __name__ == "__main__":
    main()
