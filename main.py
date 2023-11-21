import taichi as ti
import argparse
from scene import Scene
from der import Simulator

if __name__=="__main__":
    ti.init(ti.gpu, default_fp=ti.f32, debug=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scene', type=str, help='XML scene file to load')
    parser.add_argument('-o', '--outfile', type=str, help='Readable file to save simulation state to')
    parser.add_argument('-d', '--display', type=bool, help='Run the simulation with rendering enabled or disabled')
    parser.add_argument('-g', '--generate', type=bool, help='Generate PNG or not')
    args = parser.parse_args()

    scene = Scene()
    scene.load_scene(args.scene)
    scene.initialize()

    time_step = 1e-3
    framerate = 10000
    duration = 1

    sim = Simulator(scene.n_rods, scene.n_vertices, scene.params, time_step)
    sim.initialize(scene.x, scene.is_fixed, scene.v)

    frames = 0
    file = open('outfile.txt', 'w')
    sim.write_to_file(file, frames)

    window = ti.ui.Window("Hair DER", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))

    while window.running and frames < duration*framerate:
        for _ in range(int(1e-3//time_step)):
            # sim.explicit_integrator()
            sim.semi_implicit_integrator()
        frames+=1
        sim.write_to_file(file, frames)

        scene.update(sim.x)

        canvas.scene(scene.scene)
        if args.generate:
            window.save_image('output/{}.png'.format(frames))
        window.show()
    file.close()