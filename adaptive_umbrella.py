#!/usr/bin/env python3
import numpy as np
import unittest
from copy import deepcopy

#%%
class UmbrellaRunner():

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
        if len(idx) == len(lambdas):
            return tuple(idx)
        else: # if len differs, theres no index for every dimension
            raise ValueError("{} has no index.".format(lambdas))

    def _get_root_frames(self, pmf, E_max):
        """ returns the index of all positions in the pmf where the energy is
        smaller E_max"""
        selection = np.where((pmf <= E_max) & (pmf >= 0))
        zipped = list(zip(*selection))
        return zipped

    def _get_new_frames(self, pmf, root_frames):
        """ returns a dict of all frames surrounding the root_frames
        that have not an assigned energy yet, as well as their corresponding root
        frame in the format {new_frame1: root_frame1, new_frame2: root_frame2} """

        def generate_neighbor_list(root, coords=[]):
            """ recursively builds a list of all direct neighbors of the root coordinate """
            if len(coords) > 0 and len(coords[0]) == len(root):
                return [tuple(x) for x in coords]
            elif len(coords) == 0:
                coords.append([root[0]-1])
                coords.append([root[0]])
                coords.append([root[0]+1])
                return generate_neighbor_list(root, coords)
            else:
                new_coords = []
                for coord in coords:
                    dimen = len(coord)

                    new_coord = deepcopy(coord)
                    new_coord.append(root[dimen]-1)
                    new_coords.append(new_coord)

                    new_coord = deepcopy(coord)
                    new_coord.append(root[dimen])
                    new_coords.append(new_coord)

                    new_coord = deepcopy(coord)
                    new_coord.append(root[dimen]+1)
                    new_coords.append(new_coord)
                return generate_neighbor_list(root, new_coords)

        def in_pmf(frame):
            num_dimens = len(self.pmf.shape)
            for dimen in range(num_dimens):
                if frame[dimen] < 0 or frame[dimen] >= self.pmf.shape[dimen]:
                    return False
            return True


        # find all neighboring frames and create a dict that associates them to the root frame with lowest energy
        new_frames = {}
        for frame in root_frames:
            neighbors = generate_neighbor_list(frame)

            # remove neighbors that are not inside the pmf
            neighbors = [n for n in neighbors if in_pmf(n)]

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


        # remove already sampled frames (where energy >= 0)
        new_frames_list = list(new_frames.keys())
        for idx in range(len(new_frames_list)):
            new_frame = new_frames_list[idx]
            energy = pmf[new_frame]
            if energy >= 0:
                del(new_frames[new_frame])
                
        return new_frames
    
    
    def _main(self):
        # get the initial simulation and surrounding frames
        root_frames = [self._get_index_for_lambdas(self.lambda_init)]
        new_frames = self._get_new_frames(self.pmf, root_frames)
        
        self.num_iterations = 0
        
        # outer main loop: increase E and calculate PMF until E > E_max
        while True:
            
            # stop if max iterations is reached
            self.num_iterations += 1
            if(self.max_iterations > 0 and self.num_iterations > self.max_iterations):
                print("Max iterations reached ({})".format(self.max_iterations))
                return
            
            self.E = self.E_min
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Iteration: {} (max={})".format(self.num_iterations, self.max_iterations))
            new_lambdas = [self._get_lambdas_for_index(x) for x in new_frames.keys()]
            
            print("Running simulations")
            self.simulate_frames(new_frames, new_lambdas)
            
            print("Calculating new PMF")
            self.pmf = self.wham()
            self.after_run_hook()
            
            while self.E <= self.E_max:
                root_frames = self._get_root_frames(self.pmf, self.E)
                new_frames = self._get_new_frames(self.pmf, root_frames)
                
                if len(new_frames) == 0:
                    self.E += self.E_incr
                    print("Max energy increased to {} (max={})".format(self.E, self.E_max))
                else:
                    break
            

    def run(self):
        self.pmf = self._init_pmf()
        self._main()
        print("Umbrella sampling finished.")
        
    def simulate_frames(self, new_frames, new_lambdas):
        print("TODO Implement me")

    def wham(self):
        print("TODO Implement me")
    
    def after_run_hook(self):
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
        runner.pmf[0, 3] = 5
        runner.pmf[0, 2] = 2
        root_frames = runner._get_root_frames(runner.pmf, 3)
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
        runner.pmf[0, 3, 3] = 5
        runner.pmf[0, 2, 2] = 2
        root_frames = runner._get_root_frames(runner.pmf, 3)
        self.assertEquals(1, len(root_frames))
        self.assertEquals((0, 2, 2), root_frames[0])

    def test_get_new_frames(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-3, 3, 1),
            (0, 4, 1)
        ])
        runner.pmf = runner._init_pmf()
        runner.pmf[0, 3] = 5
        runner.pmf[0, 2] = 2
        runner.pmf[0, 4] = 2
        root_frames = [(0, 3), (0, 2), (0, 4)]
        new_frames = runner._get_new_frames(runner.pmf, root_frames)

        expected_new_frames = {
            (0, 1): (0, 2),
            (1, 1): (0, 2),
            (1, 2): (0, 2),
            (1, 3): (0, 2),
            (1, 4): (0, 4)
        }
        self.assertEquals(len(expected_new_frames.keys()), len(new_frames.keys()))
        self.assertDictEqual(expected_new_frames, new_frames)

    def test_get_new_frames_3d(self):
        runner = UmbrellaRunner()
        runner.cvs = np.array([
            (-1, 1, 1),
            (-1, 1, 1),
            (-1, 1, 1)
        ])
        runner.pmf = runner._init_pmf()
        runner.pmf[1, 1, 1] = 5
        root_frames = [(1, 1, 1)]
        new_frames = runner._get_new_frames(runner.pmf, root_frames)

        expected_new_frames = {}
        for x in [0, 1, 2]:
            for y in [0, 1, 2]:
                for z in [0, 1, 2]:
                    if (x, y, z) != (1, 1, 1):
                        expected_new_frames[(x, y, z)] = (1, 1, 1)

        self.assertEquals(26, len(new_frames.keys()))
        self.assertDictEqual(expected_new_frames, new_frames)


if __name__ == '__main__':
    unittest.main()


