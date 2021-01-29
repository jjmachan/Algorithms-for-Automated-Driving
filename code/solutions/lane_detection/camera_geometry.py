import numpy as np

def get_intrinsic_matrix(field_of_view_deg, image_width, image_height):
    # For our Carla camera alpha_u = alpha_v = alpha
    # alpha can be computed given the cameras field of view via
    field_of_view_rad = field_of_view_deg * np.pi/180
    alpha = (image_width / 2.0) / np.tan(field_of_view_rad / 2.)
    Cu = image_width / 2.0
    Cv = image_height / 2.0
    return np.array([[alpha, 0, Cu],
                     [0, alpha, Cv],
                     [0, 0, 1.0]])

def project_polyline(polyline_world, trafo_world_to_cam, K):
    x,y,z = polyline_world[:,0], polyline_world[:,1], polyline_world[:,2]
    homvec = np.stack((x,y,z,np.ones_like(x)))
    proj_mat = K @ trafo_world_to_cam[:3,:]
    pl_uv_cam = (proj_mat @ homvec).T
 vehicle pitch estimation:
        # https://refubium.fu-berlin.de/handle/fub188/26792
        uv_hom = np.array([u,v,1])
        Kinv_uv_hom = self.inverse_intrinsic_matrix @ uv_hom
        denominator = self.road_normal_camframe.dot(Kinv_uv_hom)
        return self.height*Kinv_uv_hom/denominator

    def uv_to_roadXYZ_roadframe(self,u,v):
        r_camframe = self.uv_to_roadXYZ_camframe(u,v)
        return self.camframe_to_roadframe(r_camframe)

    def uv_to_roadXYZ_roadframe_iso8855(self,u,v):
        X,Y,Z = self.uv_to_roadXYZ_roadframe(u,v)
        return np.array([Z,-X,-Y]) # read book section on coordinate systems to understand this

    def precompute_grid(self,dist=60):
        cut_v = int(self.compute_minimum_v(dist=dist)+1)
        xy = []
        for v in range(cut_v, self.image_height):
            for u in range(self.image_width):
                X,Y,Z= self.uv_to_roadXYZ_roadframe_iso8855(u,v)
                xy.append(np.array([X,Y]))
        xy = np.array(xy)
        return cut_v, xy

    def compute_minimum_v(self, dist):
        """
        Find cut_v such that pixels with v<cut_v are irrelevant for polynomial fitting.
        Everything that is further than `dist` along the road is considered irrelevant.
        """
        trafo_road_to_cam = np.linalg.inv(self.trafo_cam_to_road)
        point_far_away_on_road = trafo_road_to_cam @ np.array([0,0,dist,1])
        uv_vec = self.intrinsic_matrix @ point_far_away_on_road[:3]
        uv_vec /= uv_vec[2]
        cut_v = uv_vec[1]
        return cut_v
