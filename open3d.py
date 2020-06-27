import numpy as np
import open3d as o3d

def lod_mesh_export(mesh, lods, extension, path):
    mesh_lods={}
    for i in lods:
        mesh_lod = mesh.simplify_quadric_decimation(i)
        o3d.io.write_triangle_mesh(path+"lod_"+str(i)+extension, mesh_lod)
        mesh_lods[i]=mesh_lod
        o3d.visualization.draw_geometries([mesh_lod],window_name = 'Level of Details='+str(i))
    print("generation of "+str(i)+" LoD successful")
    return mesh_lods

point_cloud = np.loadtxt('pointcloud.txt.ply',skiprows=13)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,6:9]/255)
pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,3:6])

o3d.visualization.draw_geometries([pcd],window_name = 'Our patches from matching')

poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]

bbox = pcd.get_axis_aligned_bounding_box()
o3d.visualization.draw_geometries([poisson_mesh],window_name = 'Poisson Surface Reconstruction')
p_mesh_crop = poisson_mesh.crop(bbox)
o3d.visualization.draw_geometries([p_mesh_crop],window_name = 'Bounding Volume Crop')
my_lods = lod_mesh_export(p_mesh_crop, [100000,50000,10000,1000,100], ".obj", 'output_path')
# print(np.asarray(bbox.get_box_points()))
