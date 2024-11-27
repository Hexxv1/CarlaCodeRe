#!/usr/bin/env python3
import glob
import os
import sys
import random
import weakref
# Path to the .egg file
carla_egg_path = '/mnt/ml_drive/carla_project/carla/PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg'

# Add the .egg file to sys.path
if carla_egg_path not in sys.path:
    sys.path.append(carla_egg_path)
import carla

os.environ['SDL_AUDIODRIVER'] = 'dummy'
import pygame
import numpy as np

# Make sure the import statement for `sys` is before any usage of sys.version_info
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("CARLA Python API .egg file not found. Please check the path.")
    sys.exit(1)  # Exit if the CARLA .egg file cannot be found

# Continue with your script...


from pygame.locals import KMOD_CTRL, KMOD_SHIFT, K_0, K_9, K_BACKQUOTE, K_BACKSPACE, K_COMMA, K_DOWN, K_ESCAPE, K_F1, K_LEFT, K_PERIOD, K_RIGHT, K_SLASH, K_SPACE, K_TAB, K_UP, K_a, K_b, K_c, K_d, K_f, K_g, K_h, K_i, K_l, K_m, K_n, K_o, K_p, K_q, K_r, K_s, K_t, K_v, K_w, K_x, K_z, K_MINUS, K_EQUALS

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

class World:
    def __init__(self, carla_world):
        self.world = carla_world
        self.map = self.world.get_map()
        self.camera_manager = None
        self.gamma = 2.2
        self.player = None
        self.restart()

    def restart(self):
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        blueprint = self.world.get_blueprint_library().filter('cybertruck')[0]

        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 20
            spawn_point.location.roll = 0.0
            spawn_point.location.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points in the map')

            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        self.camera_manager = CameraManager(self.player, self.gamma)
        self.camera_manager.set_sensor(cam_index)

    def render(self, display):
        self.camera_manager.render(display)

    def destroy(self):
        sensors = [self.camera_manager.sensor]  # Corrected variable name
        for x in sensors:
            if x is not None:
                x.stop()
                x.destroy()
        if self.player is not None:
            self.player.destroy()

    def destroy_sensors(self):  # Corrected method name from 'destory'
        if self.camera_manager.sensor is not None:
            self.camera_manager.sensor.destroy()
            self.camera_manager.sensor = None
            self.camera_manager.index = None

class KeyboardControl:
    def __init__(self, world):
        self.set_autopilot = False
        self.control = carla.VehicleControl()
        self.steer_cache = 0.0

    def parse_events(self, client, world, clock):
        keys = pygame.key.get_pressed()  # This line was missing, causing undefined variable 'keys'
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()

            if isinstance(self.control, carla.VehicleControl):  # Corrected class name
                self.parse_vehicle_keys(keys, clock.get_time())

            if world.player is not None:
                world.player.apply_control(self.control)  # Corrected method call

    def parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self.control.throttle = min(self.control.throttle + 0.01, 1)
        else:
            self.control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self.control.brake = min(self.control.brake + 0.2, 1)
        else:
            self.control.brake = 0.0

        print('milliseconds:', milliseconds)  # Improved print statement format
        steer_increment = 5e-4 * milliseconds  # Added time factor to increment

        if keys[K_LEFT] or keys[K_a]:
            self.steer_cache -= steer_increment if self.steer_cache > -0.7 else 0
        elif keys[K_RIGHT] or keys[K_d]:
            self.steer_cache += steer_increment if self.steer_cache < 0.7 else 0
        else:
            self.steer_cache = 0.0

        self.control.steer = round(self.steer_cache, 1)

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and (pygame.key.get_mods() & KMOD_CTRL))  # Added parentheses for clarity and correctness

class CameraManager:
    def __init__(self, parent_actor, gamma_correction):
        self.parent = parent_actor
        self.gamma = gamma_correction
        Attachment = carla.AttachmentType

        self.transform_index = 0
        self.camera_transforms = [(carla.Transform(carla.Location(x=1.5, z=2.4)), Attachment.Rigid)]

        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}]
        ]

        self.index = None
        self.set_sensor(0)  # Ensure a sensor is set at initialization

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))

        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.sensor = None

            blueprint = self.parent.world.get_blueprint_library().find(self.sensors[index][0])
            if blueprint.has_attribute('image_size_x'):
                blueprint.set_attribute('image_size_x', str(1280))
                blueprint.set_attribute('image_size_y', str(720))
            if blueprint.has_attribute('gamma'):
                blueprint.set_attribute('gamma', str(self.gamma))

            spawn_point = self.camera_transforms[self.transform_index][0]
            attachment_type = self.camera_transforms[self.transform_index][1]
            self.sensor = self.parent.world.spawn_actor(blueprint, spawn_point, attach_to=self.parent, attachment_type=attachment_type)
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if image.frame % 60 == 0:  # This condition can help reduce update rate for demonstration
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Exclude alpha for RGB conversion
            array = array[:, :, ::-1]  # BGR to RGB
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

def game_loop():
    try:
        pygame.init()
        pygame.font.init()

        client = carla.Client('localhost', 1985)  # Corrected syntax for assignment
        client.set_timeout(10.0)

        display = pygame.display.set_mode((1280, 720), pygame.HWSURFACE | pygame.DOUBLEBUF)  # Corrected function name and parameters

        clock = pygame.time.Clock()

        world = World(client.get_world())

        controller = KeyboardControl(world)

        while True:
            if controller.parse_events(client, world, clock):
                return
            world.render(display)
            pygame.display.flip()
    finally:
        pygame.quit()

game_loop()
