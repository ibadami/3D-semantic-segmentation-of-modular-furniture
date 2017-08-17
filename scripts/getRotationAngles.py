import bpy
import math

print("Mesh Rotation Angles")

mesh = bpy.data.objects["Mesh"]
print(mesh.rotation_mode)

rotationZ = mesh.rotation_euler[2]
print( round( math.degrees(rotationZ) ,0 ))

rotationY = mesh.rotation_euler[1]
print( round( math.degrees(rotationY) ,0 ))

rotationX = mesh.rotation_euler[0]
print( round( math.degrees(rotationX) ,0 ))





print("Camera Rotation Angles")

camera = bpy.data.objects["Camera"]
print(camera.rotation_mode)

rotationZcam = camera.rotation_euler[2]
print( round( math.degrees(rotationZcam) ,0 ))

rotationYcam = camera.rotation_euler[1]
print( round( math.degrees(rotationYcam) ,0 ))

rotationXcam = camera.rotation_euler[0]
print( round( math.degrees(rotationXcam) ,0 ))