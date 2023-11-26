import taichi as ti
import taichi.math as tm
import numpy as np
import sys
import test

@ti.data_oriented
class Simulator:
    def __init__(self, n_rods, n_vertices, params, dt, default_fp) -> None:
        self.n_rods = n_rods
        self.n_vertices = n_vertices
        self.r = params.r #cm
        self.E = params.E #dPa = g cm^-1 s^-2
        self.G = params.G #dPa = g cm^-1 s^-2
        self.rho = params.rho # g cm^-3
        self.gravity = params.gravity # cm s^-2
        self.dt = dt
        self.default_fp = default_fp

        self.ks = np.pi* self.r**2 * self.E
        self.kt = self.G * np.pi * self.r**4 / 4

        #strand states
        self.rest_length = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
        self.rest_voronoi_length = ti.field(dtype=float, shape=(n_rods, n_vertices))
        self.rest_kappa = ti.Vector.field(4, dtype=float, shape=(n_rods, n_vertices - 2))
        self.rest_twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 2))
        self.ref_twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 2)) # referential twist

        self.x = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices)) # position cm
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
        self.grad_kappa = ti.Matrix.field(11, 4, dtype=float, shape=(n_rods, n_vertices - 2))
        self.theta = ti.field(dtype=float, shape=(n_rods, n_vertices - 1)) # Turning angle between reference frame and material frame
        self.omega = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
        self.twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 2)) # Discrete integrated twist
        self.grad_twist = ti.Vector.field(11, dtype=float, shape=(n_rods, n_vertices - 2))
        self.f_strech = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices))
        self.f_bend = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices))
        self.f_twist = ti.Vector.field(3, dtype=float, shape=(n_rods, n_vertices))
        self.tau_bend = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))
        self.tau_twist = ti.field(dtype=float, shape=(n_rods, n_vertices - 1))

        self.j_strech = ti.Matrix.field(3, 3, dtype=float, shape=(n_rods, n_vertices-1))
        self.j_bend = ti.Matrix.field(11, 11, dtype=float, shape=(n_rods, n_vertices-2))
        self.j_twist = ti.Matrix.field(11, 11, dtype=float, shape=(n_rods, n_vertices-2))
        self.vel_1D = ti.ndarray(float, n_rods*(4*n_vertices-1))
        self.force_1D = ti.ndarray(float, n_rods*(4*n_vertices-1))
        self.b = ti.ndarray(float, n_rods*(4*n_vertices-1))
        self.mass = ti.ndarray(float, n_rods*(4*n_vertices-1))
        self.A_builder = ti.linalg.SparseMatrixBuilder(n_rods * (4 * n_vertices - 1), n_rods * (4 * n_vertices - 1), max_num_triplets=20000, dtype=default_fp)
        # self.M = np.zeros((n_rods * (4 * n_vertices - 1), n_rods * (4 * n_vertices - 1)))
        # self.H = np.zeros((n_rods * (4 * n_vertices - 1), n_rods * (4 * n_vertices - 1)))

    @ti.kernel
    def compute_streching_force(self):
        for i, j in self.f_strech:
            self.f_strech[i, j] = ti.Vector([0, 0, 0])
        for i, j in self.length:
            f = self.ks *(self.length[i, j]/self.rest_length[i, j] - 1.) * self.tangent[i, j]
            # test.streching_force_test(f, self.x[i, j], self.x[i, j+1], self.ks, self.rest_length[i, j])
            self.f_strech[i, j] += f
            self.f_strech[i, j+1] -= f

    @ti.kernel
    def compute_streching_jacobian(self):
        for i, j in self.length:
            rest_length = self.rest_length[i, j]
            length = self.length[i, j]
            edge = self.tangent[i, j]
            # h = self.ks / rest_length * (edge.outer_product(edge))
            h = self.ks*((1.0 / rest_length - 1.0 / length) * ti.Matrix.identity(float,3) + 1.0 / length * (edge.outer_product(edge)))
            self.j_strech[i, j] = h

    @ti.kernel
    def compute_bending_force(self):
        for i, j in self.f_bend:
            self.f_bend[i, j] = ti.Vector([0, 0, 0])
        for i, j in self.tau_bend:
            self.tau_bend[i, j] = 0
        for i, j in self.kappa:
            b = self.E * np.pi * self.r**4 / 4
            kappa_bar = self.rest_kappa[i, j]
            ilen = 1 / self.rest_voronoi_length[i, j+1]
            f = - 0.5 * b * ilen * self.grad_kappa[i, j] @ (self.kappa[i, j] - kappa_bar)
            # test.bending_force_test(f, self.x[i, j], self.x[i, j+1], self.x[i, j+2], \
            #                 self.theta[i, j], self.theta[i, j+1], kappa_bar, \
            #                 self.n1_ref[i, j], self.n1_ref[i, j+1], self.n2_ref[i, j], self.n2_ref[i, j+1], \
            #                 b, self.rest_voronoi_length[i, j+1])
            self.f_bend[i, j] += ti.Vector([f[0], f[1], f[2]])
            self.f_bend[i, j+1] += ti.Vector([f[4], f[5], f[6]])
            self.f_bend[i, j+2] += ti.Vector([f[8], f[9], f[10]])
            self.tau_bend[i, j] += f[3]
            self.tau_bend[i, j+1] += f[7]

    @ti.kernel
    def compute_bending_jacobian(self):
        for i, j in self.kappa:
            b = self.E * np.pi * self.r**4 / 4
            ilen = 1 / self.rest_voronoi_length[i, j+1]
            h = -0.5 * b * ilen * self.grad_kappa[i, j] @ self.grad_kappa[i, j].transpose()
            h = (h+h.transpose())/2
            self.j_bend[i, j] = h

    @ti.kernel
    def compute_twisting_force(self):
        for i, j in self.f_twist:
            self.f_twist[i, j] = ti.Vector([0, 0, 0])
        for i, j in self.tau_twist:
            self.tau_twist[i, j] = 0
        for i, j in self.twist:
            twist_bar = self.rest_twist[i, j]
            ilen = 1 / self.rest_voronoi_length[i, j+1]
            f = -self.kt * ilen * self.grad_twist[i, j] * (self.twist[i, j] - twist_bar)
            # test.twisting_force_test(f, self.x[i, j], self.x[i, j+1], self.x[i, j+2], 
            #                          self.theta[i, j], self.theta[i, j+1], self.ref_twist[i, j], 
            #                          self.rest_twist[i, j], self.kt, self.rest_voronoi_length[i, j+1])
            self.f_twist[i, j] += ti.Vector([f[0], f[1], f[2]])
            self.f_twist[i, j+1] += ti.Vector([f[4], f[5], f[6]])
            self.f_twist[i, j+2] += ti.Vector([f[8], f[9], f[10]])
            self.tau_twist[i, j] += f[3]
            self.tau_twist[i, j+1] += f[7]

    @ti.kernel
    def compute_twisting_jacobian(self):
        for i, j in self.twist:
            ilen = 1 / self.rest_voronoi_length[i, j+1]
            h = -self.kt * ilen * self.grad_twist[i, j].outer_product(self.grad_twist[i, j])
            self.j_twist[i, j] = h

    @ti.kernel
    def assemble_Hessian(self, h: ti.types.sparse_matrix_builder(), mass: ti.types.ndarray()):
        for i in range(self.n_rods*(4*self.n_vertices-1)):
            h[i, i] += mass[i]
        for i in range(self.n_rods):
            for j in range(self.n_vertices-1):
                for k in range(3):
                    for l in range(3):
                        h[i*(4*self.n_vertices-1)+j*4+k, i*(4*self.n_vertices-1)+j*4+l] += self.j_strech[i, j][k, l] * self.dt**2
                        h[i*(4*self.n_vertices-1)+(j+1)*4+k, i*(4*self.n_vertices-1)+j*4+l] += -self.j_strech[i, j][k, l] * self.dt**2
                        h[i*(4*self.n_vertices-1)+j*4+k, i*(4*self.n_vertices-1)+(j+1)*4+l] += -self.j_strech[i, j][k, l] * self.dt**2
                        h[i*(4*self.n_vertices-1)+(j+1)*4+k, i*(4*self.n_vertices-1)+(j+1)*4+l] += self.j_strech[i, j][k, l] * self.dt**2
        for i in range(self.n_rods):
            for j in range(self.n_vertices-2):
                for k in range(11):
                    for l in range(11):
                        h[i*(4*self.n_vertices-1)+j*4+k, i*(4*self.n_vertices-1)+j*4+l] += -(self.j_bend[i, j][k, l] + self.j_twist[i, j][k, l])*self.dt**2

    # def assemble_Hessian(self):
    #     self.H.fill(0)
    #     for i in range(self.n_rods):
    #         for j in range(self.n_vertices-1):
    #             for k in range(3):
    #                 for l in range(3):
    #                     self.H[i*(4*self.n_vertices-1)+j*4+k, i*(4*self.n_vertices-1)+j*4+l] += -self.j_strech[i, j][k, l]
    #                     self.H[i*(4*self.n_vertices-1)+(j+1)*4+k, i*(4*self.n_vertices-1)+j*4+l] += self.j_strech[i, j][k, l]
    #                     self.H[i*(4*self.n_vertices-1)+j*4+k, i*(4*self.n_vertices-1)+(j+1)*4+l] += self.j_strech[i, j][k, l]
    #                     self.H[i*(4*self.n_vertices-1)+(j+1)*4+k, i*(4*self.n_vertices-1)+(j+1)*4+l] += -self.j_strech[i, j][k, l]
    #     for i in range(self.n_rods):
    #         for j in range(self.n_vertices-2):
    #             for k in range(11):
    #                 for l in range(11):
    #                     self.H[i*(4*self.n_vertices-1)+j*4+k, i*(4*self.n_vertices-1)+j*4+l] += self.j_bend[i, j][k, l] + self.j_twist[i, j][k, l]

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
            self.kappa[i, j] = ti.Vector([kb.dot(m2e), -kb.dot(m1e), kb.dot(m2f), -kb.dot(m1f)])

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
            chi = 1.0 + te.dot(tf)
            if chi <= 0:
                print("chi = {}, te = {}, tf = {}".format(chi, te, tf))
                chi = 1e-12
            tilde_t = (te + tf) / chi
            tilde_d1e = (2.0 * m1e) / chi
            tilde_d1f = (2.0 * m1f) / chi
            tilde_d2e = (2.0 * m2e) / chi
            tilde_d2f = (2.0 * m2f) / chi
            k = self.kappa[i, j]
            Dkappa0eDe = 1.0 / norm_e * (-k[0] * tilde_t + tf.cross(tilde_d2e))
            Dkappa0eDf = 1.0 / norm_f * (-k[0] * tilde_t - te.cross(tilde_d2e))
            Dkappa1eDe = 1.0 / norm_e * (-k[1] * tilde_t - tf.cross(tilde_d1e))
            Dkappa1eDf = 1.0 / norm_f * (-k[1] * tilde_t + te.cross(tilde_d1e))
            Dkappa0fDe = 1.0 / norm_e * (-k[2] * tilde_t + tf.cross(tilde_d2f))
            Dkappa0fDf = 1.0 / norm_f * (-k[2] * tilde_t - te.cross(tilde_d2f))
            Dkappa1fDe = 1.0 / norm_e * (-k[3] * tilde_t - tf.cross(tilde_d1f))
            Dkappa1fDf = 1.0 / norm_f * (-k[3] * tilde_t + te.cross(tilde_d1f))
            for k in ti.static(range(3)):
                self.grad_kappa[i, j][k, 0] = -Dkappa0eDe[k]
                self.grad_kappa[i, j][4+k, 0] = Dkappa0eDe[k]-Dkappa0eDf[k]
                self.grad_kappa[i, j][8+k, 0] = Dkappa0eDf[k]
                self.grad_kappa[i, j][k, 1] = -Dkappa1eDe[k]
                self.grad_kappa[i, j][4+k, 1] = Dkappa1eDe[k]-Dkappa1eDf[k]
                self.grad_kappa[i, j][8+k, 1] = Dkappa1eDf[k]
                self.grad_kappa[i, j][k, 2] = -Dkappa0fDe[k]
                self.grad_kappa[i, j][4+k, 2] = Dkappa0fDe[k]-Dkappa0fDf[k]
                self.grad_kappa[i, j][8+k, 2] = Dkappa0fDf[k]
                self.grad_kappa[i, j][k, 3] = -Dkappa1fDe[k]
                self.grad_kappa[i, j][4+k, 3] = Dkappa1fDe[k]-Dkappa1fDf[k]
                self.grad_kappa[i, j][8+k, 3] = Dkappa1fDf[k]
            kb = self.curvature_binormal[i, j]
            self.grad_kappa[i, j][3, 0] = -kb.dot(m1e)
            self.grad_kappa[i, j][7, 0] = 0.
            self.grad_kappa[i, j][3, 1] = -kb.dot(m2e)
            self.grad_kappa[i, j][7, 1] = 0.
            self.grad_kappa[i, j][3, 2] = 0.
            self.grad_kappa[i, j][7, 2] = -kb.dot(m1f)
            self.grad_kappa[i, j][3, 3] = 0.
            self.grad_kappa[i, j][7, 3] = -kb.dot(m2f)

    #https://math.stackexchange.com/questions/1143354/numerically-stable-method-for-angle-between-3d-vectors/1782769
    @ti.func
    def signed_angle(self, u, v, n):
        w = u.cross(v)
        # angle = tm.atan2(w.norm(), u.dot(v))
        angle = 2*tm.atan2((u-v).norm(), (u+v).norm())
        ret = angle
        if n.dot(w) < 0:
            ret = -angle
        return ret

    @ti.func
    def rotateAxisAngle(self, v, z, theta):
        c = tm.cos(theta)
        s = tm.sin(theta)
        return c * v + s * z.cross(v) #+ z.dot(v) * (1.0 - c) * z

    @ti.kernel
    def update_twist(self):
        for i, j in self.twist:
            u0 = self.n1_ref[i, j]
            u1 = self.n1_ref[i, j+1]
            tangent = self.tangent[i, j+1]
            ut = self.parallelTransport(u0, self.tangent[i, j], tangent)
            before_twist = self.ref_twist[i, j]
            ut = self.rotateAxisAngle(ut, tangent, before_twist)
            self.ref_twist[i, j] = before_twist + self.signed_angle(ut, u1, tangent)
            self.twist[i, j] = self.theta[i, j+1] - self.theta[i, j] + self.ref_twist[i, j]

    @ti.kernel
    def update_gradtwist(self):
        for i, j in self.kappa:
            kb = self.curvature_binormal[i, j]
            d0 = -0.5 / self.length[i, j] * kb
            d1 = 0.5 / self.length[i, j+1] * kb
            d2 = -(d0 + d1) 
            for k in ti.static(range(3)):
                self.grad_twist[i, j][0+k] = d0[k]
                self.grad_twist[i, j][8+k] = d1[k]
                self.grad_twist[i, j][4+k] = d2[k]
            self.grad_twist[i, j][3] = -1
            self.grad_twist[i, j][7] = 1

    @ti.kernel
    def update_velocity(self):
        for i, j in self.v:
            mass = self.rho * np.pi * self.r**2 * self.rest_voronoi_length[i, j]
            force = self.f_strech[i, j] + self.f_bend[i, j] + self.f_twist[i, j]
            self.v[i, j] += self.dt * force / mass

    @ti.kernel
    def add_gravity(self):
        for i, j in self.v:
            self.v[i, j] += self.dt * self.gravity

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
    def copy_to_1D(self, vel: ti.types.ndarray(), force: ti.types.ndarray()):
        for i in range(self.n_rods):
            vel[i*(4*self.n_vertices-1)]   = self.v[i, 0][0]
            vel[i*(4*self.n_vertices-1)+1] = self.v[i, 0][1]
            vel[i*(4*self.n_vertices-1)+2] = self.v[i, 0][2]
            force[i*(4*self.n_vertices-1)]   = self.f_strech[i, 0][0]+self.f_bend[i, 0][0]+self.f_twist[i, 0][0]
            force[i*(4*self.n_vertices-1)+1] = self.f_strech[i, 0][1]+self.f_bend[i, 0][1]+self.f_twist[i, 0][1]
            force[i*(4*self.n_vertices-1)+2] = self.f_strech[i, 0][2]+self.f_bend[i, 0][2]+self.f_twist[i, 0][2]
            for j in range(1, self.n_vertices):
                vel[i*(4*self.n_vertices-1)+j*4-1] = self.omega[i, j-1]
                vel[i*(4*self.n_vertices-1)+j*4]   = self.v[i, j][0]
                vel[i*(4*self.n_vertices-1)+j*4+1] = self.v[i, j][1]
                vel[i*(4*self.n_vertices-1)+j*4+2] = self.v[i, j][2]
                force[i*(4*self.n_vertices-1)+j*4-1] = self.tau_bend[i, j-1] + self.tau_twist[i, j-1]
                force[i*(4*self.n_vertices-1)+j*4]   = self.f_strech[i, j][0]+self.f_bend[i, j][0]+self.f_twist[i, j][0]
                force[i*(4*self.n_vertices-1)+j*4+1] = self.f_strech[i, j][1]+self.f_bend[i, j][1]+self.f_twist[i, j][1]
                force[i*(4*self.n_vertices-1)+j*4+2] = self.f_strech[i, j][2]+self.f_bend[i, j][2]+self.f_twist[i, j][2]

    @ti.kernel
    def add_gravity_1D(self, f: ti.types.ndarray()):
        for i in range(self.n_rods):
            for j in range(self.n_vertices):
                mass = self.rho * np.pi * self.r**2 * self.rest_voronoi_length[i, j]
                f[i*(4*self.n_vertices-1)+j*4] += mass * self.gravity[0]
                f[i*(4*self.n_vertices-1)+j*4+1] += mass * self.gravity[1]
                f[i*(4*self.n_vertices-1)+j*4+2] += mass * self.gravity[2]

    @ti.kernel
    def compute_b(self, b: ti.types.ndarray(), m: ti.types.ndarray(), v: ti.types.ndarray(), f: ti.types.ndarray()):
        for i in range(self.n_rods*(4*self.n_vertices-1)):
            b[i] = m[i]*v[i] + self.dt * f[i]

    @ti.kernel
    def update_vel(self, dv: ti.types.ndarray()):
        for i in range(self.n_rods):
            for j in range(self.n_vertices-1):
                self.v[i, j][0] = dv[i*(4*self.n_vertices-1)+j*4]
                self.v[i, j][1] = dv[i*(4*self.n_vertices-1)+j*4+1]
                self.v[i, j][2] = dv[i*(4*self.n_vertices-1)+j*4+2]
                self.omega[i, j] = dv[i*(4*self.n_vertices-1)+j*4+3]
            self.v[i, self.n_vertices-1][0] = dv[i*(4*self.n_vertices-1)+(self.n_vertices-1)*4]
            self.v[i, self.n_vertices-1][1] = dv[i*(4*self.n_vertices-1)+(self.n_vertices-1)*4+1]
            self.v[i, self.n_vertices-1][2] = dv[i*(4*self.n_vertices-1)+(self.n_vertices-1)*4+2]

    def semi_implicit_integrator(self):
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
        self.compute_streching_jacobian()
        self.compute_bending_jacobian()
        self.compute_twisting_jacobian()
        self.assemble_Hessian(self.A_builder, self.mass)
        A = self.A_builder.build()
        self.copy_to_1D(self.vel_1D, self.force_1D)
        self.add_gravity_1D(self.force_1D)
        self.compute_b(self.b, self.mass, self.vel_1D, self.force_1D)
        solver = ti.linalg.SparseSolver(solver_type="LLT", dtype=self.default_fp)
        solver.analyze_pattern(A)
        solver.factorize(A)
        dv = solver.solve(self.b)
        # assert solver.info()

        # self.assemble_Hessian()
        # self.copy_to_1D(self.vel_1D, self.force_1D)
        # self.add_gravity_1D(self.force_1D)
        # A = self.M - self.dt**2 * self.H
        # Mv = self.M @ self.vel_1D.to_numpy()
        # self.compute_b(self.b, Mv, self.force_1D)
        # dv = np.linalg.solve(A, self.b.to_numpy())
        self.update_vel(dv)

    @ti.kernel
    def init_mass(self, m: ti.types.ndarray(dtype=float, ndim=1)):
        for i in range(self.n_rods):
            for j in range(self.n_vertices):
                mass = self.rho * np.pi * self.r**2 * self.rest_voronoi_length[i, j]
                m[i*(4*self.n_vertices-1)+j*4  ] = mass
                m[i*(4*self.n_vertices-1)+j*4+1] = mass
                m[i*(4*self.n_vertices-1)+j*4+2] = mass
        for i in range(self.n_rods):
            for j in range(self.n_vertices-1):
                mass = self.rho * np.pi * self.r**2 * self.rest_length[i, j]
                m[i*(4*self.n_vertices-1)+j*4+3] = 0.5 * mass * self.r**2

    # def init_mass_matrix(self):
    #     for i in range(self.n_rods):
    #         for j in range(self.n_vertices):
    #             mass = self.rho * np.pi * self.r**2 * self.rest_voronoi_length[i, j]
    #             self.M[i*(4*self.n_vertices-1)+j*4  , i*(4*self.n_vertices-1)+j*4  ] = mass
    #             self.M[i*(4*self.n_vertices-1)+j*4+1, i*(4*self.n_vertices-1)+j*4+1] = mass
    #             self.M[i*(4*self.n_vertices-1)+j*4+2, i*(4*self.n_vertices-1)+j*4+2] = mass
    #     for i in range(self.n_rods):
    #         for j in range(self.n_vertices-1):
    #             mass = self.rho * np.pi * self.r**2 * self.rest_length[i, j]
    #             self.M[i*(4*self.n_vertices-1)+j*4+3, i*(4*self.n_vertices-1)+j*4+3] = 0.5 * mass * self.r**2


    @ti.kernel
    def init_reference_frame(self):
        for i in ti.static(range(self.n_rods)):
            self.n1_ref[i, 0] = ti.Vector([-self.tangent[i, 0][1], self.tangent[i, 0][0], 0])
            self.n2_ref[i, 0] = tm.cross(self.tangent[i, 0], self.n1_ref[i, 0])
        for i in ti.static(range(self.n_rods)):
            for j in ti.static(range(1, self.n_vertices - 1)):
                self.n1_ref[i, j] = self.parallelTransport(self.n1_ref[i, j-1], self.tangent[i, j-1], self.tangent[i, j])
                self.n2_ref[i, j] = self.parallelTransport(self.n2_ref[i, j-1], self.tangent[i, j-1], self.tangent[i, j])
        for i, j in self.n1_ref:
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
        self.init_mass(self.mass)
        # self.init_mass_matrix()
        self.init_reference_frame()
        self.update_curvature_binormal()
        self.update_kappa()
        for i in ti.static(range(self.n_rods)):
            for j in ti.static(range(self.n_vertices - 2)):
                self.rest_kappa[i, j] = self.kappa[i, j]
        self.update_twist()
        for i in ti.static(range(self.n_rods)):
            for j in ti.static(range(self.n_vertices - 2)):
                self.rest_twist[i, j] = self.twist[i, j]

    def write_to_file(self, outfile, frame):
        outfile.write('------frame {}-----\n'.format(frame))
        outfile.write('position:\n{}\n'.format(self.x))
        outfile.write('theta:\n{}\n'.format(self.theta))
        outfile.write('velocity:\n{}\n'.format(self.v))
        outfile.write('omega:\n{}\n'.format(self.omega))
        outfile.write('twist:\n{}\n'.format(self.twist))
        outfile.write('reference twist:\n{}\n'.format(self.ref_twist))
        # outfile.write('streching force:\n{}\n'.format(self.f_strech))
        # outfile.write('bending force:\n{}\n'.format(self.f_bend))
        # outfile.write('bending torque:\n{}\n'.format(self.tau_bend))
        # outfile.write('twisting force:\n{}\n'.format(self.f_twist))
        # outfile.write('twisting torque:\n{}\n'.format(self.tau_twist))
        # outfile.write('curvature binormal:\n{}\n'.format(self.curvature_binormal))
        # outfile.write('reference frame1:\n{}\n'.format(self.n1_ref))
        # outfile.write('reference frame2:\n{}\n'.format(self.n2_ref))
        # outfile.write('material frame1:\n{}\n'.format(self.n1_mat))
        # outfile.write('material frame2:\n{}\n'.format(self.n2_mat))
