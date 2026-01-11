import bpy
import os

ply_path = "C:/Users/John/Desktop/plyfile.ply"
glb_path = "C:/Users/John/Desktop/glbfolder.glb"

ply_unix = ply_path.replace('\\', '/')
glb_unix = glb_path.replace('\\', '/')
script = (
    "import bpy; "
    f"bpy.ops.import_scene.ply(filepath='{ply_unix}'); "
    f"bpy.ops.export_scene.gltf(filepath='{glb_unix}', export_format='GLB')"
)

bpy.ops.import_scene.ply(filepath=ply_unix)
bpy.ops.export_scene.gltf(filepath=glb_unix, export_format='GLB')