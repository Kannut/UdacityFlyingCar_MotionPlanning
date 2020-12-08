import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

import matplotlib.pyplot as plt

from bresenham import bresenham

from planning_utils import a_star, heuristic, create_grid, create_grid_and_edges, create_graph, closest_point, heuristic_g, a_star_g, prune_path
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection, grid, north_offset, east_offset, edges, target_altitude, goal_coordinates):
        super().__init__(connection)

        self.grid = grid
        self.north_offset = north_offset
        self.east_offset = east_offset
        self.edges = edges
        self.target_altitude = target_altitude
        self.goal_coordinates = goal_coordinates

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")        

        self.target_position[2] = self.target_altitude

        # DONE: read lat0, lon0 from colliders into floating point values
        with open('colliders.csv') as f:
            lat_lon_line = f.readline()
            lat_lon_list = lat_lon_line.split(", ")

            for d in lat_lon_list:
                if d.startswith("lat0"):
                    lat0 = float(d.split(" ")[1])

                if d.startswith("lon0"):
                    lon0 = float(d.split(" ")[1])

        # DONE: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)

        # DONE: retrieve current global position
        global_position_current = self.global_position
 
        # DONE: convert to current local position using global_to_local()
        local_position_current = global_to_local(global_position_current, self.global_home)
        
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
                

        print("North offset = {0}, east offset = {1}".format(self.north_offset, self.east_offset))
        # Define starting point on the grid (this is just grid center)
        grid_start = (-self.north_offset, -self.east_offset)
        # DONE: convert start position to current position rather than map center
        grid_start = (grid_start[0]+int(local_position_current[0]),grid_start[1]+int(local_position_current[1]))
        
        # Set goal as some arbitrary position on the grid
        # grid_goal = (-north_offset + 10, -east_offset + 10)
        # DONE: adapt to set goal as latitude / longitude position and convert
        #goal_coordinates = (-122.396810, 37.795108, 0)
        #goal_coordinates = (-122.397650, 37.792780, 0)
        goal_coordinates_local = global_to_local(self.goal_coordinates, self.global_home)
        grid_goal = (-self.north_offset+int(goal_coordinates_local[0]),-self.east_offset+int(goal_coordinates_local[1]))
       

        # Run A* to find a path from start to goal
        # DONE: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
      
        
        # Check if graph or grid based
        if len(edges) != 0:
            graph = create_graph(self.edges)
            start_ne_g = closest_point(graph, grid_start)
            goal_ne_g = closest_point(graph, grid_goal)
            print('Local Start and Goal: ', start_ne_g, goal_ne_g)
            path, _ = a_star_g(graph, heuristic_g, start_ne_g, goal_ne_g)
        else:
            print('Local Start and Goal: ', grid_start, grid_goal)
            path, _ = a_star(grid, heuristic, grid_start, grid_goal)


        # Plotting
        '''
        plt.imshow(self.grid, origin='lower', cmap='Greys') 

        for e in edges:
            p1 = e[0]
            p2 = e[1]
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-')

        for i in range(len(path)-1):
            p1 = path[i]
            p2 = path[i+1]
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-')
        plt.plot([goal_ne_g[1], goal_ne_g[1]], [goal_ne_g[0], goal_ne_g[0]], 'r-')

        plt.plot(grid_start[1], grid_start[0], 'gx')
        plt.plot(grid_goal[1], grid_goal[0], 'gx')

        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        plt.show()
        '''

        # DONE: prune path to minimize number of waypoints - only done for the grid based A* method
        print("Length of path before pruning: ", len(path))
        path = prune_path(path, 1)
        print("Length of path after pruning: ", len(path))
        

        # DONE (if you're feeling ambitious): Try a different approach altogether!

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        # DONE: send waypoints to sim (this is just for visualization of waypoints) - COMMENTED OUT DUE TO TAKEOFF TRANSISTION NOT WORKING
        #self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()


    TARGET_ALTITUDE = 5
    SAFETY_DISTANCE = 2
    # Read in obstacle map
    data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
    # Define a grid for a particular altitude and safety margin around obstacles
    # grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
    # edges = []

    grid, north_offset, east_offset, edges = create_grid_and_edges(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
    print('Found %5d edges' % len(edges))

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=10*60)
    drone = MotionPlanning(conn, grid, north_offset, east_offset, edges,TARGET_ALTITUDE, (-122.396810, 37.795108, 0))
    time.sleep(1)
    drone.start()
