#!/usr/bin/env python3
import numpy as np
import unittest
from copy import deepcopy

#%%
class UmbrellaRunner():
    """
    Runner class for adaptive umbrella sampling
    
    Attributes:
        cvs (numpy array): A multidimensional array of all collective variables (cvs) in the form [cv_min, cv_max, cv_delta]
         Example: to sample a 2 dimensional grid where cv1 and cv2 go from 0 to 3 with stepsize 0.5, use 
         np.array([[0,3,1],[0,3,1]])
        cvs_init (tuple): Starting coordinates for umbrella sampling. Must be a tuple with the same dimension as cvs
        E_min (float, default=0): Starting energy for exporative umbrella sampling
        E_max (float, default=inf): Final energy. Umbrella sampling is stopped if no frames with E < E_max are found
        E_incr (float, default=1): E_min is incremented by this until E_max is reached
        max_iterations (int, default=-1): Max. number of iterations before umbrella sampling stops. -1 for infinite sampling
        reset_E (boolean, default=False): Wether the energy should be reset to E_min at the start of each cycle
        
    """

    def __init__(self):
        self.max_iterations = -1
        self.E_min = 0
        self.E_max = np.inf
        self.E_incr = 1
        self.reset_E = False

    def _get_pmf_shape(self):
        """ returns the shape of the pmf according to the cvs """
        shape = []
        for dimen in self.cvs:
            windows = np.arange(*dimen)
            size = len(windows)
            if windows[-1] % dimen[-1] == 0:
                size += 1
            shape.append(size)
        return shape


    def _init_pmf(self):
        """ returns an empty matrix where each dimension equals the number of frames along the corresponding reaction
        coordinate """
        shape = self._get_pmf_shape()
        pmf = np.empty(shape)
        pmf.fill(-1)
        return pmf
        
    def _get_lambdas_for_index(self, idx):
        """ takes a coordinate tuple of the pmf and returns corresponding lambda values """

        lambdas = self.cvs.T[0] + idx * self.cvs.T[2]
        return tuple(np.round(lambdas, 10))

    def _get_index_for_lambdas(self, lambdas):
        """ takes a lambda tuple and returns corresponding indexes of the pmf 
        TODO: faster implementation required
        """
        idx = []
        for dimen in range(len(lambdas)):
            cv = self.cvs[dimen]
            r = np.arange(cv[0], cv[1]+cv[2], cv[2])
        
            for i in range(len(r)):
                if abs(r[i]-lambdas[dimen]) < 0.00001:
                    idx.append(i)
                    break
            if not len(idx)-1 == dimen:
                raise ValueError(f"{lambdas} has no index because {lambdas[dimen]} is not in {r}.")

        return tuple(idx)
        
    def _get_sampled_lambdas(self, step=None):
        """ returns an array of all sampled lambdas. If step is given, only return
        lambdas for this step"""
        if step is None:
            sampled_coords = np.where(self.sample_list > 0)
        else:
            sampled_coords = np.where(self.sample_list == step)
        sampled_coords = np.array(sampled_coords).T
        
        return np.array([self._get_lambdas_for_index(cvs) for cvs in sampled_coords])

    def _get_root_frames(self, pmf, frames, E_max):
        """ returns the index of all positions in the pmf where the energy is
        smaller E_max"""

        # select positions of the pmf where E <= E_max and that have already been sampled (frames > 0)
        selection = np.where((pmf <= E_max) & (frames > 0))
        zipped = list(zip(*selection))

        return zipped

    def _generate_neighbor_list(self, root):
        """ builds a list of all direct neighbors of the root coordinate """
        dimens = len(root)
        coords = []
        for i in range(dimens):
            if len(coords) == 0:
                new_coords = []
                new_coords.append([root[i]-1])
                new_coords.append([root[i]])
                new_coords.append([root[i]+1])
                coords = new_coords
            else:
                new_coords = []
                for coord in coords:
                    new_coords.append(coord + [root[i] - 1])
                    new_coords.append(coord + [root[i]])
                    new_coords.append(coord + [root[i] + 1])
                coords = new_coords
        return [tuple(x) for x in coords]


    def _is_in_pmf(self, frame):
        num_dimens = len(self.pmf.shape)
        for dimen in range(num_dimens):
            if frame[dimen] < 0 or frame[dimen] >= self.pmf.shape[dimen]:
                return False
        return True

    def is_valid_frame(self, frame):
        """ Allows to filter out frames that should not be sampled depending
        on some condition """
        # do not allow frames that are not inside pmf boundaries
        return self._is_in_pmf(frame)

    def _get_new_frames(self, pmf, frames, root_frames):
        """ returns a dict of all frames surrounding the root_frames
        that have not an assigned energy yet, as well as their corresponding root
        frame in the format {new_frame1: root_frame1, new_frame2: root_frame2} """

        # find all neighboring frames and create a dict that associates them to the root frame with lowest energy
        new_frames = {}
        for frame in root_frames:
            neighbors = self._generate_neighbor_list(frame)

            # remove neighbors if they are not valid (i.e not part of the pmf)
            neighbors = [n for n in neighbors if self.is_valid_frame(n)]

            # for each neighbor, check if its already in the list and compare root frame energy
            for n in neighbors:
                try:
                    root_energy = pmf[frame]
                    old_root = new_frames[n]
                    old_root_energy = pmf[old_root]
                    if root_energy < old_root_energy:
                        new_frames[n] = frame
                except KeyError:
                    new_frames[n] = frame


        # remove already sampled frames (frames > 0)
        new_frames_list = list(new_frames.keys())
        for idx in range(len(new_frames_list)):
            new_frame = new_frames_list[idx]
            if frames[new_frame] > 0:
                del(new_frames[new_frame])
                
        return new_frames
    
    
    def _main(self):
        
        self.num_iterations = 0

        self.E = self.E_min
        
        # outer main loop: increase E and calculate PMF until E > E_max
        while True:
            self.num_iterations += 1
            if self.reset_E:
                self.E = self.E_min
            
            print("~~~~~~~~~~~~~~~ Iteration {}/{} ~~~~~~~~~~~~~~~~".format(self.num_iterations, self.max_iterations))
            print("Energy: {}".format(self.E))


            # find frames to sample
            if self.num_iterations == 1:
                # get the initial simulation and surrounding frames
                root_frames = [self._get_index_for_lambdas(self.cvs_init)]
                new_frames = self._get_new_frames(self.pmf, self.sample_list, root_frames)
            else:
                while self.E <= self.E_max:
                    # get new frames for energy level
                    root_frames = self._get_root_frames(self.pmf, self.sample_list, self.E)
                    new_frames = self._get_new_frames(self.pmf, self.sample_list, root_frames)
                    
                    # increase energy level if no frames are found
                    if len(new_frames) == 0:
                        self.E += self.E_incr
                        print("Max energy increased to {} (max={})".format(self.E, self.E_max))
                    else:
                        break

            # abort sampling if no new frames could be found
            if len(new_frames) == 0:
                print("Sampling aborted. No neighbors found for E_max={}".format(self.E))
                return


            # lambda states for new frames
            lambdas = dict([(self._get_lambdas_for_index(x), self._get_lambdas_for_index(y)) for x,y in new_frames.items()])

            self.pre_run_hook()

            print("Running simulations")
            self.simulate_frames(lambdas, new_frames)
    
            # update list of sampled windows
            for new_frame in new_frames.keys():
                self.sample_list[new_frame] = self.num_iterations


            print("Calculating new PMF")
            self.pmf = self.calculate_new_pmf()

            self.after_run_hook()

            if(self.max_iterations > 0 and self.num_iterations == self.max_iterations):
                print("Max iterations reached ({})".format(self.max_iterations))
                return

            # stop if the pmf is sampled
            if len(np.where(self.pmf < 0)) == 0:
                print("Every window of the PMF appears to be sampled.")
                return


    def run(self):
        # initialize the pmf
        self.pmf = self._init_pmf()

        # a list of sampled frames of the pmf (0=unsampled, 1,2,3.. = sampled )
        self.sample_list = np.zeros(self.pmf.shape)

        # start the simulation/evaluation loop
        self._main()
        print("Finished.")
        
    def simulate_frames(self, frames, lambdas):
        """ This should be implemented to run the simulation for each frame in ```frames```
         with ```lambdas```. Both variables are dictionaries where new values are keys and root values are values
         """
        print("New lambda values to simulate: {}".format(lambdas))
        print("(You should implement `simulate_frames` method yourself.)")

    def calculate_new_pmf(self):
        """ This should be overwritten to calculate the PMF based on the new simulations and
        return a pmf of correct shape/spacing according to the cvs"""
        pass


    def after_run_hook(self):
        """ This can be implemented to hook into the simulation cycle after the reevaluation of the pmf """
        pass

    def pre_run_hook(self):
        """ This can be implemented to hook into the simulation cycle before the simulation runs """
        pass

class UmbrellaRunnerTest(unittest.TestCase):

    def test_init_pmf_3d(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (-3, 3, 1),
            (-3, 2, 1)
        ])
        pmf = runner._init_pmf()

        expected_shape = (7, 7, 6)
        self.assertEquals(pmf.shape, expected_shape)

    def test_init_pmf_odd(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (-3.5, 3, 1)
        ])
        pmf = runner._init_pmf()
        expected_shape = (7, 7)
        self.assertEquals(pmf.shape, expected_shape)

    def test_get_lambdas_for_index(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (0, 4, 1)
        ])

        lambdas = runner._get_lambdas_for_index((0, 0))
        self.assertAlmostEqual(lambdas, (-3, 0))

        lambdas = runner._get_lambdas_for_index((3, 2))
        self.assertEquals(lambdas, (0, 2))

    def test_get_index_for_lambdas(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (0, 4, 1)
        ])

        index = runner._get_index_for_lambdas((3, 3))
        self.assertEquals(index, (6, 3))

    def test_get_index_for_lambdas_error(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (0, 4, 1)
        ])

        with self.assertRaises(ValueError) as error:
            runner._get_index_for_lambdas((3, 2.5))

    def test_get_root_frames(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (0, 4, 1)
        ])
        runner.pmf = runner._init_pmf()
        runner.sample_list = np.zeros(runner.pmf.shape)

        runner.pmf[0, 3] = 5
        runner.pmf[0, 2] = 2
        runner.pmf[0, 1] = 2
        runner.sample_list[0, 3] = 1
        runner.sample_list[0, 2] = 1
        root_frames = runner._get_root_frames(runner.pmf, runner.sample_list, 3)
        self.assertEquals(1, len(root_frames))
        self.assertEquals((0, 2), root_frames[0])

    def test_get_root_frames_3d(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (0, 4, 1),
            (0, 4, 1)
        ])
        runner.pmf = runner._init_pmf()
        runner.sample_list = np.zeros(runner.pmf.shape)
        runner.pmf[0, 3, 3] = 5
        runner.pmf[0, 2, 2] = 2
        runner.pmf[0, 1, 2] = 2
        runner.sample_list[0, 3, 3] = 5
        runner.sample_list[0, 2, 2] = 2
        root_frames = runner._get_root_frames(runner.pmf, runner.sample_list, 3)
        self.assertEquals(1, len(root_frames))
        self.assertEquals((0, 2, 2), root_frames[0])

    def test_generate_neighbor_list(self):
        runner = UmbrellaRunner()
        root = (1, 3)
        neighbors = runner._generate_neighbor_list(root)
        expected_neighbors = [
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 2),
            (2, 3),
            (2, 4),
        ]
        self.assertEquals(neighbors, expected_neighbors)

    def test_generate_neighbor_list2(self):
        runner = UmbrellaRunner()
        root = (3, 1)
        neighbors = runner._generate_neighbor_list(root)
        expected_neighbors = [
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 0),
            (3, 1),
            (3, 2),
            (4, 0),
            (4, 1),
            (4, 2),
        ]
        self.assertEquals(neighbors, expected_neighbors)
        pass

    def test_generate_neighbor_list_3d(self):
        runner = UmbrellaRunner()
        root = (2, 2, 2)
        neighbors = runner._generate_neighbor_list(root)
        expected_neighbors = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    expected_neighbors.append((root[0]+x, root[1]+y, root[2]+z))
        self.assertEquals(len(neighbors), len(expected_neighbors))


    def test_is_in_pmf(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-1, 1, 1),
            (-1, 1, 1),
            (-1, 1, 1)
        ])
        runner.pmf = runner._init_pmf()
        self.assertFalse(runner._is_in_pmf((-1, 0, 0)))
        self.assertFalse(runner._is_in_pmf((0, 0, 3)))
        self.assertTrue(runner._is_in_pmf((0, 0, 0)))

    def test_get_new_frames(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-2, 2, 1),
            (-2, 2, 1),
        ])
        runner.pmf = np.array([
            [-1, -1, -1, -1, -1],
            [-1, 34, 32, 26, -1],
            [-1, 40, 42, 51, -1],
            [-1, 28, 41, 37, -1],
            [-1, -1, -1, -1, -1]])
        runner.sample_list = np.zeros(runner.pmf.shape)
        runner.sample_list[runner.pmf > 0] = 1
        runner.sample_list[runner.pmf < 0] = 0
        root_frames = [(1, 3), (3, 1)]
        expected_new_frames = {
            (0, 2): (1, 3),
            (0, 3): (1, 3),
            (0, 4): (1, 3),
            (1, 4): (1, 3),
            (2, 4): (1, 3),
            (2, 0): (3, 1),
            (3, 0): (3, 1),
            (4, 0): (3, 1),
            (4, 1): (3, 1),
            (4, 2): (3, 1),
        }
        new_frames = runner._get_new_frames(runner.pmf, runner.sample_list, root_frames)
        self.assertDictEqual(expected_new_frames, new_frames)

    def test_get_new_frames_differing_pmf(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-2, 2, 1),
            (-2, 2, 1),
        ])
        runner.pmf = np.array([
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, -1, 30, -1, -1],
            [-1, -1, 20, -1, -1],
            [-1, -1, -1, -1, -1]])
        runner.sample_list = np.zeros(runner.pmf.shape)
        runner.sample_list[2, 3] = 1
        root_frames = [(2, 3)]
        expected_new_frames = {
            (1, 2): (2, 3),
            (2, 2): (2, 3),
            (3, 2): (2, 3),
            (1, 3): (2, 3),
            (3, 3): (2, 3),
            (1, 4): (2, 3),
            (2, 4): (2, 3),
            (3, 4): (2, 3)
        }
        new_frames = runner._get_new_frames(runner.pmf, runner.sample_list, root_frames)
        self.assertDictEqual(expected_new_frames, new_frames)


if __name__ == '__main__':
    unittest.main()


