import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow/")
from MMFlow import transform, MMFlow, utils
import openmm.unit as unit

class NNForceNet(nn.Module):
    def __init__(self,
                 n_angles,
                 n_dihedrals,
                 n_distances,
                 hidden_size):
        super(NNForceNet, self).__init__()

        self.n_angles = n_angles
        self.n_dihedrals = n_dihedrals
        self.n_distances = n_distances        
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(n_angles + 2*n_dihedrals + n_distances,
                             hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)        
        self.fc4 = nn.Linear(hidden_size, 1, bias = False)        

    def forward(self, angles, dihedrals, distances):
        x = torch.cat((torch.cos(angles),
                       torch.cos(dihedrals), torch.sin(dihedrals),
                       distances), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))     
        output = self.fc4(x)
        output = torch.squeeze(output)
        return output

class NNForce(nn.Module):
    def __init__(self,
                 n_particles,
                 hidden_size,
                 p_index_for_angle,
                 p_index_for_dihedral,
                 p_index_for_distance,
                 T):
        super(NNForce, self).__init__()
        self.n_particles = n_particles
        self.hidden_size = hidden_size
        self.p_index_for_angle = nn.Parameter(p_index_for_angle,
                                              requires_grad = False)
        self.p_index_for_dihedral = nn.Parameter(p_index_for_dihedral,
                                                 requires_grad = False)
        self.p_index_for_distance = nn.Parameter(p_index_for_distance,
                                                 requires_grad = False)
        
        self.kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.kelvin * unit.AVOGADRO_CONSTANT_NA
        self.kbT = self.kbT.value_in_unit(unit.kilojoule_per_mole)
        
        self.nnforce_net = NNForceNet(p_index_for_angle.shape[0],
                                      p_index_for_dihedral.shape[0],
                                      p_index_for_distance.shape[0],
                                      hidden_size)
        
    def forward(self, positions):
        xyz = torch.unsqueeze(positions, 0)
        angles = utils.functional.compute_angles(xyz, self.p_index_for_angle)        
        dihedrals = utils.functional.compute_dihedrals(xyz, self.p_index_for_dihedral)
        distances = utils.functional.compute_distances(xyz, self.p_index_for_distance)
        potential_energy = self.kbT * self.nnforce_net(angles, dihedrals, distances)
        return potential_energy    
