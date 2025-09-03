import numpy as np
import openmm as omm
import openmm.unit as unit
import openmm.app as ommapp
import math
import mdtraj
import pickle
from sys import exit
import sys
sys.path.append("/home/gridsan/dingxq/my_package_on_github/MMFlow")
from MMFlow import utils
import torch

def make_system(masses,
                coor_transformer,
                bonded_parameters,
                T,
                exclude_angle_particle_index = [],
                exclude_dihedral_particle_index = [],
                jacobian = False):
    
    ## add particles
    system = omm.System()
    for m in masses:
        system.addParticle(m)

    Kb = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
    Kb = Kb.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)

    ## bond force between reference particle 2 and reference particle 1
    bond_force_ref_2 = omm.CustomBondForce("0.5*kb*(r - b0)^2*Kb*T")
    bond_force_ref_2.addGlobalParameter("Kb", Kb)
    bond_force_ref_2.addGlobalParameter("T", T)
    bond_force_ref_2.addPerBondParameter('b0')
    bond_force_ref_2.addPerBondParameter('kb')
    bond_force_ref_2.addBond(
        coor_transformer.ref_particle_1,
        coor_transformer.ref_particle_2,
        [bonded_parameters['reference_particle_2_bond']['b0'],
         bonded_parameters['reference_particle_2_bond']['kb']]
    )
    
    bond_force_ref_2.setForceGroup(0)
    system.addForce(bond_force_ref_2)

    ## bond force between reference particle 3 and reference particle 1
    if jacobian:
        bond_force_ref_3 = omm.CustomBondForce("(0.5*kb*(r - b0)^2 + log(r))*Kb*T")
    else:
        bond_force_ref_3 = omm.CustomBondForce("(0.5*kb*(r - b0)^2)*Kb*T")        
    bond_force_ref_3.addGlobalParameter("Kb", Kb)
    bond_force_ref_3.addGlobalParameter("T", T)
    bond_force_ref_3.addPerBondParameter('b0')
    bond_force_ref_3.addPerBondParameter('kb')
    bond_force_ref_3.addBond(
        coor_transformer.ref_particle_1,
        coor_transformer.ref_particle_3,
        [bonded_parameters['reference_particle_3_bond']['b0'],
         bonded_parameters['reference_particle_3_bond']['kb']]
    )
    bond_force_ref_3.setForceGroup(0)
    system.addForce(bond_force_ref_3)

    ## bond force between other particles
    if jacobian:
        bond_force = omm.CustomBondForce("(0.5*kb*(r - b0)^2 + 2*log(r))*Kb*T")
    else:
        bond_force = omm.CustomBondForce("(0.5*kb*(r - b0)^2)*Kb*T")        
    bond_force.addGlobalParameter("Kb", Kb)
    bond_force.addGlobalParameter("T", T)
    bond_force.addPerBondParameter('b0')
    bond_force.addPerBondParameter('kb')
    for i in range(len(coor_transformer.particle_visited_in_order)):
        p = coor_transformer.particle_visited_in_order[i]
        p1 = coor_transformer.bond_particle_idx[p][0]
        p2 = coor_transformer.bond_particle_idx[p][1]
        bond_force.addBond(
            p1,
            p2,
            [float(bonded_parameters['bond']['b0'][i].item()),
             float(bonded_parameters['bond']['kb'][i].item())]
        )
    bond_force.setForceGroup(0)
    system.addForce(bond_force)
    
    ## custom angle force
    if 'a0' in bonded_parameters['reference_particle_3_angle'] and 'ka' in bonded_parameters['reference_particle_3_angle']:
        angle_force_between_reference_particles = omm.CustomAngleForce("0.5*ka*(theta - a0)^2*Kb*T")
        angle_force_between_reference_particles.addGlobalParameter('Kb', Kb)
        angle_force_between_reference_particles.addGlobalParameter('T', T)
        angle_force_between_reference_particles.addPerAngleParameter('a0')
        angle_force_between_reference_particles.addPerAngleParameter('ka')
        if not (coor_transformer.ref_particle_1 in exclude_angle_particle_index and
                coor_transformer.ref_particle_2 in exclude_angle_particle_index and
                coor_transformer.ref_particle_3 in exclude_angle_particle_index):
            angle_force_between_reference_particles.addAngle(
                coor_transformer.ref_particle_2,
                coor_transformer.ref_particle_1,
                coor_transformer.ref_particle_3,
                [bonded_parameters['reference_particle_3_angle']['a0'],
                 bonded_parameters['reference_particle_3_angle']['ka']]
            )
        angle_force_between_reference_particles.setForceGroup(0)
        system.addForce(angle_force_between_reference_particles)

        if jacobian:
            angle_force = omm.CustomAngleForce("(alpha*0.5*ka*(theta - a0)^2 + log(sin(pi - theta)))*Kb*T; pi=3.14159265357")
        else:
            angle_force = omm.CustomAngleForce("(alpha*0.5*ka*(theta - a0)^2)*Kb*T")
        angle_force.addGlobalParameter('Kb', Kb)
        angle_force.addGlobalParameter('T', T)
        angle_force.addPerAngleParameter('a0')
        angle_force.addPerAngleParameter('ka')
        angle_force.addPerAngleParameter('alpha')

        for pi in range(len(coor_transformer.particle_visited_in_order)):
            p = coor_transformer.particle_visited_in_order[i]
            p1 = coor_transformer.angle_particle_idx[p][0]
            p2 = coor_transformer.angle_particle_idx[p][1]
            p3 = coor_transformer.angle_particle_idx[p][2]
            if (p1 in exclude_angle_particle_index and
                p2 in exclude_angle_particle_index and
                p3 in exclude_angle_particle_index):
                alpha = 0.0
            else:
                alpha = 1.0
                
            angle_force.addAngle(
                p1,
                p2,
                p3,
                [float(bonded_parameters['angle']['a0'][i].item()),
                 float(bonded_parameters['angle']['ka'][i].item()),
                 alpha]
            )
        angle_force.setForceGroup(0)        
        system.addForce(angle_force)
        
    elif 'U' in bonded_parameters['reference_particle_3_angle']:
        ## angle force between reference particles
        f = omm.Continuous1DFunction(bonded_parameters['reference_particle_3_angle']['U'].tolist(), 0.0, np.pi, periodic = False)
        angle_force_between_reference_particles = omm.CustomCompoundBondForce(3, f"ua_between_ref(angle(p1, p2, p3))*Kb*T")
        angle_force_between_reference_particles.addGlobalParameter('Kb', Kb)
        angle_force_between_reference_particles.addGlobalParameter('T', T)
        angle_force_between_reference_particles.addTabulatedFunction(f"ua_between_ref", f)
        if not (coor_transformer.ref_particle_1 in exclude_angle_particle_index and
                coor_transformer.ref_particle_2 in exclude_angle_particle_index and
                coor_transformer.ref_particle_3 in exclude_angle_particle_index):            
            angle_force_between_reference_particles.addBond(
                [coor_transformer.ref_particle_2,
                 coor_transformer.ref_particle_1,
                 coor_transformer.ref_particle_3])
        angle_force_between_reference_particles.setForceGroup(0)
        system.addForce(angle_force_between_reference_particles)

        ## angle force between other particles        
        angle_parameters = bonded_parameters['angle']
        ua = torch.stack([angle_parameters[i]['U'] for i in range(len(angle_parameters))]).numpy()
        func = omm.Continuous2DFunction(xsize = ua.shape[0],
                                        ysize = ua.shape[1],
                                        values = ua.reshape(-1, order = 'F'),
                                        xmin = 0.0, xmax = float(ua.shape[0] - 1),
                                        ymin = 0.0, ymax = np.pi,
                                        periodic = False)
        if jacobian:
            angle_force = omm.CustomCompoundBondForce(
                3,
                f"(alpha*ua(idx, angle(p1, p2, p3)) + log(sin(pi - angle(p1, p2, p3))))*Kb*T; pi = 3.14159265357"
            )
        else:
            angle_force = omm.CustomCompoundBondForce(
                3,
                f"(alpha*ua(idx, angle(p1, p2, p3)))*Kb*T"
            )
            
        angle_force.addGlobalParameter('Kb', Kb)
        angle_force.addGlobalParameter('T', T)
        angle_force.addTabulatedFunction("ua", func)
        angle_force.addPerBondParameter('idx')
        angle_force.addPerBondParameter('alpha')
        
        for i in range(len(coor_transformer.particle_visited_in_order)):
            p = coor_transformer.particle_visited_in_order[i]
            p1, p2, p3 = coor_transformer.angle_particle_idx[p]
            if (p1 in exclude_angle_particle_index and
                p2 in exclude_angle_particle_index and
                p3 in exclude_angle_particle_index):
                alpha = 0.0
            else:
                alpha = 1.0
            angle_force.addBond([p1, p2, p3], [float(i), alpha])            
        angle_force.setForceGroup(0)
        system.addForce(angle_force)
        
    ## torsion force
    dihedral_parameters = bonded_parameters['dihedral']
    ud = torch.stack([dihedral_parameters[i]['U'] for i in range(len(dihedral_parameters))] +
                     [dihedral_parameters[0]['U']]).numpy()
    func = omm.Continuous2DFunction(xsize = ud.shape[0],
                                    ysize = ud.shape[1],
                                    values = ud.reshape(-1, order = 'F'),
                                    xmin = 0.0, xmax = float(ud.shape[0] - 1),
                                    ymin = -np.pi, ymax = np.pi,
                                    periodic = True)
    dihedral_force = omm.CustomCompoundBondForce(4, f"(alpha*ud(idx, dihedral(p1, p2, p3, p4)))*Kb*T")
    dihedral_force.addGlobalParameter('Kb', Kb)
    dihedral_force.addGlobalParameter('T', T)
    dihedral_force.addTabulatedFunction("ud", func)
    dihedral_force.addPerBondParameter('idx')
    dihedral_force.addPerBondParameter('alpha')    

    for i in range(len(coor_transformer.particle_visited_in_order)):
        p = coor_transformer.particle_visited_in_order[i]
        p1, p2, p3, p4 = coor_transformer.dihedral_particle_idx[p]
        if (p1 in exclude_dihedral_particle_index and
            p2 in exclude_dihedral_particle_index and
            p3 in exclude_dihedral_particle_index and                
            p4 in exclude_dihedral_particle_index):
            alpha = 0.0
        else:
            alpha = 1.0
        dihedral_force.addBond([p1, p2, p3, p4], [float(i), alpha])
    dihedral_force.setForceGroup(0)
    system.addForce(dihedral_force)

    system.addForce(omm.CMMotionRemover())
    
    return system
