import taichi as ti
import taichi.math as tm
import numpy as np
import sys

@ti.data_oriented
class Simulator:
    def __init__(self, n_rods, n_vertices, params, dt) -> None:
        self.n_rods = n_rods
        self.n_vertices = n_vertices
        self.r = params.r
        self.E = params.E
        self.G = params.G
        self.rho = params.rho
        self.dt = dt
        #strand states
        self.rest_length = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
        self.rest_voronoi_length = ti.field(dtype=float, shape=(n_rods, n_vertices))
        self.rest_kappa = ti.Vector.field(4, dtype=float, shape=(n_rods, n_vertices - 2))
        self.rest_twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 2))
        self.ref_twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 2)) # referential twist

        self.x = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices)) # position
        self.is_fixed = ti.field(dtype=bool, shape=(n_rods, n_vertices))
        self.v = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices)) # velocity
        self.edge = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
        self.length = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
        self.old_tangent = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
        self.tangent = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
        self.n1_mat = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
        self.n2_mat = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
        self.n1_ref = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
        self.n2_ref = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 1))
        self.curvature_binormal = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 2))
        self.kappa = ti.Vector.field(4, dtype=float, shape=(n_rods, n_vertices - 2))
        self.grad_kappa = ti.Matrix.field(4, 3, dtype=float, shape=(n_rods, n_vertices - 2, 3))
        self.grad_theta_kappa = ti.Vector.field(4, dtype=float, shape=(n_rods, n_vertices - 2, 2))
        self.theta = ti.field(dtype=float, shape=(n_rods, n_vertices - 1)) # Turning angle between reference frame and material frame
        self.omega = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
        self.twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 2)) # Discrete integrated twist
        self.grad_twist = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices - 2, 3))
        self.f_strech = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices))
        self.f_bend = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices))
        self.f_twist = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices))
        self.tau_bend = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
        self.tau_twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))

    @ti.kernel
    def compute_streching_force(self):
        for i, j in self.f_strech:
            self.f_strech[i, j] = [0, 0, 0]
        for i, j in self.length:
            f = np.pi* self.r**2 * self.E *(self.length[i, j]/self.rest_length[i, j] - 1) * self.tangent[i, j]
            self.f_strech[i, j] += f
            self.f_strech[i, j+1] -= f

    @ti.kernel
    def compute_bending_force(self):
        for i, j in self.f_bend:
            self.f_bend[i, j] = [0, 0, 0]
        for i, j in self.tau_bend:
            self.tau_bend[i, j] = 0
        for i, j in self.kappa:
            b = self.E * np.pi * self.r**4 / 8
            kappa_bar = self.rest_kappa[i, j]
            ilen = 1 / self.rest_voronoi_length[i, j+1]
            self.f_bend[i, j] += -b*ilen*self.grad_kappa[i, j, 0].transpose()@(self.kappa[i, j] - kappa_bar)
            self.f_bend[i, j+1] += -b*ilen*self.grad_kappa[i, j, 1].transpose()@(self.kappa[i, j] - kappa_bar)
            self.f_bend[i, j+2] += -b*ilen*self.grad_kappa[i, j, 2].transpose()@(self.kappa[i, j] - kappa_bar)
            self.tau_bend[i, j] += -b*ilen * self.grad_theta_kappa[i, j, 0].dot(self.kappa[i, j] - kappa_bar)
            self.tau_bend[i, j+1] += -b*ilen * self.grad_theta_kappa[i, j, 1].dot(self.kappa[i, j] - kappa_bar)

    @ti.kernel
    def compute_twisting_force(self):
        for i, j in self.f_twist:
            self.f_twist[i, j] = [0, 0, 0]
        for i, j in self.tau_twist:
            self.tau_twist[i, j] = 0
        for i, j in self.twist:
            b = self.G * np.pi * self.r**4 / 4
            twist_bar = self.rest_twist[i, j]
            ilen = 1 / self.rest_voronoi_length[i, j+1]
            self.f_twist[i, j] += -b*ilen*self.grad_twist[i, j, 0]*(self.twist[i, j] - twist_bar)
            self.f_twist[i, j+1] += -b*ilen*self.grad_twist[i, j, 1]*(self.twist[i, j] - twist_bar)
            self.f_twist[i, j+2] += -b*ilen*self.grad_twist[i, j, 2]*(self.twist[i, j] - twist_bar)
            self.tau_twist[i, j] += b*ilen*(self.twist[i, j] - twist_bar)
            self.tau_twist[i, j+1] += -b*ilen*(self.twist[i, j] - twist_bar)

    @ti.func
    def parallelTransport(self, n, t0, t1):
        b = t0.cross(t1)
        ret = ti.Vector([0., 0., 0.])
        if(b.norm()<sys.float_info.epsilon):
            ret = n
        else :
            b = b.normalized()
            n0 = t0.cross(b)
            n1 = t1.cross(b)
            ret = n.dot(n0) * n1 + n.dot(b) * b
        return ret

    @ti.kernel
    def restore_tangents(self):
        for i, j in self.tangent:
            self.old_tangent[i, j] = self.tangent[i, j]

    @ti.kernel
    def update_position(self):
        for i, j in self.x:
            if not self.is_fixed[i, j]:
                self.x[i, j] += self.dt * self.v[i, j]

    @ti.kernel
    def update_theta(self):
        for i, j in self.theta:
            self.theta[i, j] += self.dt * self.omega[i, j]

    @ti.kernel
    def update_edge_tangent_length(self):
        for i, j in self.edge:
            self.edge[i, j] = self.x[i, j+1] - self.x[i, j]
            self.tangent[i, j] = self.edge[i, j].normalized()
            self.length[i, j] = self.edge[i, j].norm()

    @ti.kernel
    def update_material_frame(self):
        for i, j in self.theta:
            self.n1_ref[i, j] = self.parallelTransport(self.n1_ref[i, j], self.old_tangent[i, j], self.tangent[i, j])
            self.n2_ref[i, j] = self.parallelTransport(self.n2_ref[i, j], self.old_tangent[i, j], self.tangent[i, j])
            self.n1_mat[i, j] = tm.cos(self.theta[i, j])*self.n1_ref[i, j] + tm.sin(self.theta[i, j])*self.n2_ref[i, j]
            self.n2_mat[i, j] = -tm.sin(self.theta[i, j])*self.n1_ref[i, j] + tm.cos(self.theta[i, j])*self.n2_ref[i, j]

    @ti.kernel
    def update_curvature_binormal(self):
        for i, j in self.curvature_binormal:
            t1 = self.tangent[i, j]
            t2 = self.tangent[i, j+1]
            self.curvature_binormal[i, j] = 2*t1.cross(t2)/(1+t1.dot(t2))

    @ti.kernel
    def update_kappa(self):
        for i, j in self.kappa:
            kb = self.curvature_binormal[i, j]
            m1e = self.n1_mat[i, j]
            m2e = self.n2_mat[i, j]
            m1f = self.n1_mat[i, j+1]
            m2f = self.n2_mat[i, j+1]
            self.kappa[i, j] = ti.Vector([m2e.dot(kb), m2f.dot(kb), -m1e.dot(kb), -m1f.dot(kb)])

    @ti.kernel
    def update_gradkappa(self):
        for i, j in self.kappa:
            norm_e = self.length[i, j]
            norm_f = self.length[i, j+1]
            te = self.tangent[i, j]
            tf = self.tangent[i, j+1]
            m1e = self.n1_mat[i, j]
            m2e = self.n2_mat[i, j]
            m1f = self.n1_mat[i, j+1]
            m2f = self.n2_mat[i, j+1]
            chi = 1.0 + tm.dot(te, tf)
            if chi <= 0:
                print("chi = {}, te = {}, tf = {}".format(chi, te, tf))
                chi = 1e-12
            tilde_t = (te + tf) / chi
            k = self.kappa[i, j]
            Dkappa0De = 1.0 / norm_e * (-k[0] * tilde_t + 2*tm.cross(tf, m2e / chi))
            Dkappa0Df = 1.0 / norm_f * (-k[0] * tilde_t - 2*tm.cross(te, m2e / chi))
            Dkappa1De = 1.0 / norm_e * (-k[1] * tilde_t + 2*tm.cross(tf, m2f / chi))
            Dkappa1Df = 1.0 / norm_f * (-k[1] * tilde_t - 2*tm.cross(te, m2f / chi))
            Dkappa2De = 1.0 / norm_e * (-k[2] * tilde_t - 2*tm.cross(tf, m1e / chi))
            Dkappa2Df = 1.0 / norm_f * (-k[2] * tilde_t + 2*tm.cross(te, m1e / chi))
            Dkappa3De = 1.0 / norm_e * (-k[3] * tilde_t - 2*tm.cross(tf, m1f / chi))
            Dkappa3Df = 1.0 / norm_f * (-k[3] * tilde_t + 2*tm.cross(te, m1f / chi))
            self.grad_kappa[i, j, 0] = ti.Matrix.rows([-Dkappa0De, -Dkappa1De, -Dkappa2De, -Dkappa3De])
            self.grad_kappa[i, j, 1] = ti.Matrix.rows([Dkappa0De-Dkappa0Df, Dkappa1De-Dkappa1Df, Dkappa2De-Dkappa2Df, Dkappa3De-Dkappa3Df])
            self.grad_kappa[i, j, 2] = ti.Matrix.rows([Dkappa0Df, Dkappa1Df, Dkappa2Df, Dkappa3Df])
            kb = self.curvature_binormal[i, j]
            self.grad_theta_kappa[i, j, 0] = ti.Vector([0, -tm.dot(kb, m1e), 0, -tm.dot(kb, m2e)])
            self.grad_theta_kappa[i, j, 1] = ti.Vector([-tm.dot(kb, m1f), 0, -tm.dot(kb, m2f), 0])

    @ti.func
    def signed_angle(self, u, v, n):
        w = u.cross(v)
        angle = tm.atan2(w.norm(), u.dot(v))
        ret = angle
        if n.dot(w) < 0:
            ret = -angle
        return ret

    @ti.func
    def rotateAxisAngle(self, v, z, theta):
        c = tm.cos(theta)
        s = tm.sin(theta)
        v = c * v + s * z.cross(v) + z.dot(v) * (1.0 - c) * z

    @ti.kernel
    def update_twist(self):
        for i, j in self.twist:
            v1 = self.parallelTransport(self.n1_ref[i, j], self.tangent[i, j], self.tangent[i, j+1])
            v2 = self.n1_ref[i, j+1]
            before_twist = self.ref_twist[i, j]
            self.rotateAxisAngle(v1, self.tangent[i, j+1], before_twist)
            self.ref_twist[i, j] = before_twist + self.signed_angle(v1, v2, self.tangent[i, j+1])
            self.twist[i, j] = self.theta[i, j+1] - self.theta[i, j] + self.ref_twist[i, j]

    @ti.kernel
    def update_gradtwist(self):
        for i, j in self.kappa:
            kb = self.curvature_binormal[i, j]
            self.grad_twist[i, j, 0] = -0.5 / self.length[i, j] * kb
            self.grad_twist[i, j, 2] = 0.5 / self.length[i, j+1] * kb
            self.grad_twist[i, j, 1] = -(self.grad_twist[i, j, 0] + self.grad_twist[i, j, 2])

    @ti.kernel
    def update_velocity(self):
        for i, j in self.v:
            mass = self.rho * np.pi * self.r**2 * self.rest_voronoi_length[i, j]
            force = self.f_strech[i, j] + self.f_bend[i, j] + self.f_twist[i, j]
            self.v[i, j] += self.dt * force / mass

    @ti.kernel
    def add_gravity(self):
        for i, j in self.v:
            self.v[i, j] += self.dt * ti.Vector([0, -981, 0])

    @ti.kernel
    def update_omega(self):
        for i, j in self.omega:
            mass = self.rho * np.pi * self.r**2 * self.rest_length[i, j]
            self.omega[i, j] += self.dt * (self.tau_bend[i, j] + self.tau_twist[i, j]) / (0.5 * mass * self.r**2)

    def explicit_integrator(self):
        self.restore_tangents()
        self.update_position()
        self.update_theta()
        self.update_edge_tangent_length()
        self.update_material_frame()
        self.update_curvature_binormal()
        self.update_kappa()
        self.update_gradkappa()
        self.update_twist()
        self.update_gradtwist()
        self.compute_streching_force()
        self.compute_bending_force()
        self.compute_twisting_force()
        self.update_velocity()
        self.add_gravity()
        self.update_omega()

    @ti.kernel
    def init_reference_frame(self):
        for i, j in self.n1_ref:
            self.n1_ref[i, j] = ti.Vector([-self.tangent[i, j][1], self.tangent[i, j][0], 0])
            self.n2_ref[i, j] = tm.cross(self.tangent[i, j], self.n1_ref[i, j])
            self.n1_mat[i, j] = self.n1_ref[i, j]
            self.n2_mat[i, j] = self.n2_ref[i, j]

    def initialize(self, x, is_fixed, v):
        for i in ti.static(range(self.n_rods)):
            for j in ti.static(range(self.n_vertices)):
                self.x[i, j] = x[i * self.n_vertices + j]
                self.is_fixed[i, j] = is_fixed[i * self.n_vertices + j]
                self.v[i, j] = v[i * self.n_vertices + j]
        self.update_edge_tangent_length()
        for i in ti.static(range(self.n_rods)):
            for j in ti.static(range(self.n_vertices - 1)):
                self.rest_length[i, j] = self.length[i, j]
                self.rest_voronoi_length[i, j] += self.length[i, j] / 2
                self.rest_voronoi_length[i, j+1] += self.length[i, j] / 2
        self.init_reference_frame()
        self.update_curvature_binormal()
        for i in ti.static(range(self.n_rods)):
            for j in ti.static(range(self.n_vertices - 1)):
                self.rest_twist[i, j] = 0
        self.update_kappa()
        for i in ti.static(range(self.n_rods)):
            for j in ti.static(range(self.n_vertices - 2)):
                self.rest_kappa[i, j] = self.kappa[i, j]

    def write_to_file(self, outfile, frame):
        outfile.write('------frame {}-----\n'.format(frame))
        outfile.write('position:\n{}\n'.format(self.x))
        outfile.write('velocity:\n{}\n'.format(self.v))
        outfile.write('theta:\n{}\n'.format(self.theta))
        outfile.write('twist:\n{}\n'.format(self.twist))
        outfile.write('streching force:\n{}\n'.format(self.f_strech))
        outfile.write('bending force:\n{}\n'.format(self.f_bend))
        outfile.write('twisting force:\n{}\n'.format(self.f_twist))
        outfile.write('curvature binormal:\n{}\n'.format(self.curvature_binormal))
        outfile.write('reference twist:\n{}\n'.format(self.ref_twist))
        outfile.write('reference frame1:\n{}\n'.format(self.n1_ref))
        outfile.write('reference frame2:\n{}\n'.format(self.n2_ref))
        outfile.write('material frame1:\n{}\n'.format(self.n1_mat))
        outfile.write('material frame2:\n{}\n'.format(self.n2_mat))
