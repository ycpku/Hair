import xml.etree.ElementTree as ET
import taichi as ti

class StrandParameters:
    def __init__(self, radius = 0.0037, YoungsModulus = 3.9e10,
                 shearModulus = 3.4e9, density = 1.32):
        self.r = radius
        self.E = YoungsModulus
        self.G = shearModulus
        self.rho = density
        self.gravity = ti.Vector([0,-981,0])

@ti.data_oriented
class Scene:
    def __init__(self, scene) -> None:
        self.n_rods = 0
        self.n_vertices = 0
        self.scene = scene
        self.camera = ti.ui.Camera()
        self.x = []
        self.is_fixed = []
        self.v = []
        self.params = StrandParameters()

    def initialize_strand(self, strand):
        for p in strand:
            assert(p.tag == 'particle')
            self.x.append(ti.Vector([float(x) for x in p.attrib['x'].split()]))
            self.is_fixed.append(int(p.attrib.get('fixed','0')))
            self.v.append(ti.Vector([float(x) for x in p.attrib.get('v','0 0 0').split()]))

    def set_params(self, strand_parameters):
        for p in strand_parameters:
            if p.tag == 'radius':
                self.params.r = float(p.attrib['value'])
            if p.tag == 'youngsModulus':
                self.params.E = float(p.attrib['value'])
            if p.tag == 'shearModulus':
                self.params.G = float(p.attrib['value'])
            if p.tag == 'density':
                self.params.rho = float(p.attrib['value'])
            if p.tag == 'gravity':
                self.params.gravity = ti.Vector([float(x) for x in p.attrib['value'].split()])

    def set_camera(self, camera):
        position = [float(x) for x in camera.attrib.get('position','0 -0.25 5').split()]
        lookat = [float(x) for x in camera.attrib.get('lookat','0 -0.25 0').split()]
        up = [float(x) for x in camera.attrib.get('up','0 1 0').split()]
        fov = int(camera.attrib.get('fov','45'))
        self.camera.position(position[0],position[1],position[2])
        self.camera.lookat(lookat[0],lookat[1],lookat[2])
        self.camera.up(up[0],up[1],up[2])
        self.camera.fov(fov)

    def load_scene(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        for child in root:
            if child.tag == 'Strand':
                if self.n_rods == 0:
                    self.n_vertices = len(child)
                assert(self.n_vertices == len(child))
                self.initialize_strand(child)
                self.n_rods += 1
            if child.tag == 'camera':
                self.set_camera(child)
            if child.tag == 'StrandParameters':
                self.set_params(child)
            if child.tag == 'duration':
                self.time = float(child.attrib['time'])

    def initialize(self):
        self.indices = ti.field(dtype=int, shape=2*(self.n_vertices - 1))
        for i in ti.static(range(2*(self.n_vertices-1))):
            self.indices[i] = (i+1)//2
        self.vertices = ti.Vector.field(3, dtype=float, shape=self.n_rods * self.n_vertices)

    @ti.kernel
    def update_vertices(self, x: ti.template()):
        for i, j in x:
            self.vertices[i * self.n_rods + j] = x[i, j]

    def update(self, x):
        self.update_vertices(x)
        self.scene.set_camera(self.camera)
        self.scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        self.scene.ambient_light((0.5, 0.5, 0.5))
        self.scene.particles(self.vertices, radius=0.003, color=(0, 0, 0))
        self.scene.lines(self.vertices, width=4, indices=self.indices, color=(0, 0, 0)) #TODO: multiple strands