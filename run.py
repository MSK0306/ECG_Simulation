#!/usr/bin/env python

# Modified by Kocak on 15 September 2023
import os
from datetime import date
from carputils import settings
from carputils import tools
from carputils import mesh
from carputils import ep
from carputils.carpio import txt
import numpy as np
import matplotlib.pyplot as plt

EXAMPLE_DIR = os.path.dirname(__file__)
CALLER_DIR = os.getcwd()

def parser():
    
    parser = tools.standard_parser()
    group = parser.add_argument_group('experiment specific options')
    group.add_argument('--duration',
                       type = float, default = 100.)
    group.add_argument('--sourceModel',
                       default = 'monodomain')
    group.add_argument('--tmECG',
                       type = str, default = None)
    # Simulation settings added by Kocak
    group.add_argument('--meshname',
                       default = 'empty')
    group.add_argument('--stimname',
                       default = 'empty')
    group.add_argument('--ionicmodel',
                       default = 'empty')
    group.add_argument('--conductivityfactor',
                       type = float, default = 1.)
    group.add_argument('--bathconductivity',
                       type = float, default = 1.)
    group.add_argument('--dt',
                       type = float, default = 1.)
    
    return parser

def jobID(args):

    today = date.today()
    return '{}_{}_{}_dur_{}_ms_{}_{}_cf_{}_bc_{}_dt_{}_us'.format(today.isoformat(), args.meshname, args.sourceModel, args.duration, args.stimname, args.ionicmodel, args.conductivityfactor, args.bathconductivity, args.dt)

@tools.carpexample(parser, jobID, clean_pattern='^(\d{4}-\d{2}-\d{2})|(mesh)|(.pts)')
def run(args, job):

    if args.tmECG is not None:

        compute_tmECG(args.tmECG, job)
        return
    
    # Defining the mesh
    meshname = args.meshname
    
    # Defining the tags
    tags = {'bath1': 1,
            'bath2': 2,
            'bath3': 3,
            'bath4': 4,
            'bath5': 5,
            'bath6': 6,
            'bath7': 7,
            'bath8': 8,
            'bath9': 9,
            'bath10': 10,
            'bath11': 11,
            'bath12': 12,
            'bath13': 13,
            'bath14': 14,
            'bath15': 15,
            'bath16': 16,
            'bath17': 17,
            'bath18': 18,
            'bath19': 19,
            'bath20': 20,
            'bath21': 21,
            'bath22': 22,
            'bath23': 23,
            'bath24': 24,
            'bath25': 25,
            'bath26': 26,
            'bath27': 27,
            'bath28': 28,
            'bath29': 29,
            'bath30': 30,
            'bath31': 31,
            'bath32': 32,
            'bath33': 33,
            'RV': 34, 'LV': 35}
    
    # Defining the tags for domains
    _, etags,_ = txt.read(meshname + '.elem')
    etags = np.unique(etags)
    IntraTags = [34, 35]        # element labels for extracellular grid
    ExtraTags = etags.copy()    # element labels for intracellular grid
    # Stimulation domain must be inside proper domain!!!

    # Set up ionic heterogeneity
    imp_reg = ionic_setup(tags,args)

    # Set up conductive heterogeneity
    g_reg = setup_gregions(tags, args)

    # Set up stimulation
    stimname = args.stimname
    stimfiledirectory = os.path.join(EXAMPLE_DIR, stimname)
    
    # Stimulation specifications
    stim = [
            "-num_stim", 1,
            "-stim[0].crct.type", 0,            # Stimulate using transmembrane current 
            "-stim[0].pulse.strength", 250,     # uA/cm^2
            "-stim[0].ptcl.start", 10.,         # Spply stimulus at [ms]
            "-stim[0].ptcl.duration", 2,
            "-stim[0].elec.vtx_file", stimfiledirectory
            ]
  
    # LATs, calculated in the absence of bath
    lat = setup_lats()

    # ECG, write a grid for monodomain and pseudo_bidomain extracellularpotential recoveries. Refer to the example.
    # writeECGgrid(wedgeSz, args.bath)

    # Recover phie's at given locations
    # ecg = ['-phie_rec_ptf', os.path.join(CALLER_DIR, 'ecg')]

    # Determine model type
    srcmodel = ep.model_type_opts(args.sourceModel)

    # Numerical parameters
    num_par = ['-dt',        args.dt,   # Defines the time step size to solve the numeric equations for. [us]
               '-parab_solve', 1]

    # I/O
    IO_par = ['-spacedt', args.dt/1000,      # Defines the temporal interval to output data to files. It can only be as small as 'dt/1000'. [ms]
              '-timedt',  0.4]      # Defines the temporal interval between progress updates made to the terminal.

    # Get basic command line, including solver options
    cmd = tools.carp_cmd()

    cmd += imp_reg
    cmd += g_reg
    cmd += stim
    cmd += num_par
    cmd += IO_par
    cmd += lat
    # cmd += ecg
    cmd += srcmodel
    cmd += tools.gen_physics_opts(ExtraTags=ExtraTags, IntraTags=IntraTags)

    cmd += ['-meshname', meshname,
            '-tend',     args.duration,
            '-simID',    job.ID]

    # Run simulation 
    job.carp(cmd, 'Extracellular potentials and ECGs')
    
# Local function definitions

def ionic_setup(tags,args):
    """
    Set up heterogeneities in ionic properties
    """
    imp_reg = ['-num_imp_regions',        1,
               '-imp_region[0].im',       args.ionicmodel,
               '-imp_region[0].num_IDs',  2,
               '-imp_region[0].ID[0]',    tags['RV'],
               '-imp_region[0].ID[1]',    tags['LV']]
    return imp_reg

def setup_gregions(tags, args):
    """
    Setup heterogeneities in conductivity
    """
    cf = args.conductivityfactor
    bath = args.bathconductivity
    
    g_reg = ['-num_gregions', 2,
             '-gregion[0].num_IDs', 2,
             '-gregion[0].ID[0]', tags['RV'],
             '-gregion[0].ID[1]', tags['LV'],
             # Courtemanche 1000 um, obtained with tuneCV. Refer to the example.
             '-gregion[0].g_il', cf*0.7433,
             '-gregion[0].g_it', cf*0.2981,
             '-gregion[0].g_in', cf*0.1512,
             '-gregion[0].g_el', cf*2.67,
             '-gregion[0].g_et', cf*1.0707,
             '-gregion[0].g_en', cf*0.5430,
             '-gregion[1].num_IDs', 33,
             '-gregion[1].ID[0]', tags['bath1'],
             '-gregion[1].ID[1]', tags['bath2'],
             '-gregion[1].ID[2]', tags['bath3'],
             '-gregion[1].ID[3]', tags['bath4'],
             '-gregion[1].ID[4]', tags['bath5'],
             '-gregion[1].ID[5]', tags['bath6'],
             '-gregion[1].ID[6]', tags['bath7'],
             '-gregion[1].ID[7]', tags['bath8'],
             '-gregion[1].ID[8]', tags['bath9'],
             '-gregion[1].ID[9]', tags['bath10'],
             '-gregion[1].ID[10]', tags['bath11'],
             '-gregion[1].ID[11]', tags['bath12'],
             '-gregion[1].ID[12]', tags['bath13'],
             '-gregion[1].ID[13]', tags['bath14'],
             '-gregion[1].ID[14]', tags['bath15'],
             '-gregion[1].ID[15]', tags['bath16'],
             '-gregion[1].ID[16]', tags['bath17'],
             '-gregion[1].ID[17]', tags['bath18'],
             '-gregion[1].ID[18]', tags['bath19'],
             '-gregion[1].ID[19]', tags['bath20'],
             '-gregion[1].ID[20]', tags['bath21'],
             '-gregion[1].ID[21]', tags['bath22'],
             '-gregion[1].ID[22]', tags['bath23'],
             '-gregion[1].ID[23]', tags['bath24'],
             '-gregion[1].ID[24]', tags['bath25'],
             '-gregion[1].ID[25]', tags['bath26'],
             '-gregion[1].ID[26]', tags['bath27'],
             '-gregion[1].ID[27]', tags['bath28'],
             '-gregion[1].ID[28]', tags['bath29'],
             '-gregion[1].ID[29]', tags['bath30'],
             '-gregion[1].ID[30]', tags['bath31'],
             '-gregion[1].ID[30]', tags['bath32'],
             '-gregion[1].ID[30]', tags['bath33'],
             '-gregion[1].g_bath', bath
             ]
    return g_reg

def setup_lats():
    """
    Simple setup for lat detection based on Vm and threshold crossing
    """
    LATthreshold = -10
    
    lat = ['-num_LATs',            1,
           '-lats[0].ID',      "LATs",
           '-lats[0].all',         0,
           '-lats[0].measurand',   0,
           '-lats[0].threshold', LATthreshold,
           '-lats[0].method',      1]
    return lat

# Modify ECG grid function for recovery positions!!!

def writeECGgrid():
    """
    Setup for electrode positions definitions
    """
    # Positions of the electrodes
    
    pts = np.array([0,0,0])        
    pts = np.vstack((pts, [0,0,1]))   
    pts = np.vstack((pts, [0,0,2])) 
    pts = np.vstack((pts, [0,0,3])) 
    pts = np.vstack((pts, [0,0,4])) 
    pts = np.vstack((pts, [0,0,5])) 
    pts = np.vstack((pts, [0,0,6])) 
    pts = np.vstack((pts, [0,0,7])) 
    pts = np.vstack((pts, [0,0,8])) 
    pts = np.vstack((pts, [0,0,9])) 

    txt.write(os.path.join(CALLER_DIR, 'ecg.pts'), pts)

def compute_tmECG(tmECG, job):
    """
    Extract endocardial and epicardial unipolar electrograms
    to compute the ECG.
    """

    # Extracting electrode data
    # Check the nodes for specific mesh!!!
    extract_RA = [settings.execs.igbextract, '-l', 94593,
                                               '-O', os.path.join(tmECG, 'RA.dat'),
                                               '-o', 'ascii',
                                               os.path.join(tmECG, 'phie.igb')]
    job.bash(extract_RA)

    extract_RL = [settings.execs.igbextract, '-l', 86094,
                                               '-O', os.path.join(tmECG, 'RL.dat'),
                                               '-o', 'ascii',
                                               os.path.join(tmECG, 'phie.igb')]
    job.bash(extract_RL)
    
    extract_LA = [settings.execs.igbextract, '-l', 94146,
                                               '-O', os.path.join(tmECG, 'LA.dat'),
                                               '-o', 'ascii',
                                               os.path.join(tmECG, 'phie.igb')]
    job.bash(extract_LA)
    
    extract_LL = [settings.execs.igbextract, '-l', 86173,
                                               '-O', os.path.join(tmECG, 'LL.dat'),
                                               '-o', 'ascii',
                                               os.path.join(tmECG, 'phie.igb')]
    job.bash(extract_LL)
    
    extract_V1 = [settings.execs.igbextract, '-l', 91469,
                                               '-O', os.path.join(tmECG, 'V1.dat'),
                                               '-o', 'ascii',
                                               os.path.join(tmECG, 'phie.igb')]
    job.bash(extract_V1)
    
    extract_V2 = [settings.execs.igbextract, '-l', 91481,
                                               '-O', os.path.join(tmECG, 'V2.dat'),
                                               '-o', 'ascii',
                                               os.path.join(tmECG, 'phie.igb')]
    job.bash(extract_V2)
    
    extract_V3 = [settings.execs.igbextract, '-l', 91833,
                                               '-O', os.path.join(tmECG, 'V3.dat'),
                                               '-o', 'ascii',
                                               os.path.join(tmECG, 'phie.igb')]
    job.bash(extract_V3)
    
    extract_V4 = [settings.execs.igbextract, '-l', 89976,
                                               '-O', os.path.join(tmECG, 'V4.dat'),
                                               '-o', 'ascii',
                                               os.path.join(tmECG, 'phie.igb')]
    job.bash(extract_V4)
    
    extract_V5 = [settings.execs.igbextract, '-l', 89910,
                                               '-O', os.path.join(tmECG, 'V5.dat'),
                                               '-o', 'ascii',
                                               os.path.join(tmECG, 'phie.igb')]
    job.bash(extract_V5)
    
    extract_V6 = [settings.execs.igbextract, '-l', 90331,
                                               '-O', os.path.join(tmECG, 'V6.dat'),
                                               '-o', 'ascii',
                                               os.path.join(tmECG, 'phie.igb')]
    job.bash(extract_V6)
    
    # Read the traces
    RA_trace = txt.read(os.path.join(tmECG, 'RA.dat'))
    RL_trace  = txt.read(os.path.join(tmECG, 'RL.dat'))
    LA_trace = txt.read(os.path.join(tmECG, 'LA.dat'))
    LL_trace  = txt.read(os.path.join(tmECG, 'LL.dat'))
    V1_trace  = txt.read(os.path.join(tmECG, 'V1.dat'))
    V2_trace  = txt.read(os.path.join(tmECG, 'V2.dat'))
    V3_trace  = txt.read(os.path.join(tmECG, 'V3.dat'))
    V4_trace  = txt.read(os.path.join(tmECG, 'V4.dat'))
    V5_trace  = txt.read(os.path.join(tmECG, 'V5.dat'))
    V6_trace  = txt.read(os.path.join(tmECG, 'V6.dat'))
    
    # Dump the ECGs
    WilsonCT = (LA_trace+RA_trace+LL_trace)/3
    txt.write(os.path.join(tmECG, 'Lead1.dat'), LA_trace-RA_trace)
    txt.write(os.path.join(tmECG, 'Lead2.dat'), LL_trace-RA_trace)
    txt.write(os.path.join(tmECG, 'Lead3.dat'), LL_trace-LA_trace)
    txt.write(os.path.join(tmECG, 'LeadaVR.dat'), RA_trace-0.5*(LA_trace+LL_trace))
    txt.write(os.path.join(tmECG, 'LeadaVL.dat'), LA_trace-0.5*(RA_trace+LL_trace))
    txt.write(os.path.join(tmECG, 'LeadaVF.dat'), LL_trace-0.5*(LA_trace+RA_trace))
    txt.write(os.path.join(tmECG, 'LeadV1.dat'), V1_trace-WilsonCT)
    txt.write(os.path.join(tmECG, 'LeadV2.dat'), V2_trace-WilsonCT)
    txt.write(os.path.join(tmECG, 'LeadV3.dat'), V3_trace-WilsonCT)
    txt.write(os.path.join(tmECG, 'LeadV4.dat'), V4_trace-WilsonCT)
    txt.write(os.path.join(tmECG, 'LeadV5.dat'), V5_trace-WilsonCT)
    txt.write(os.path.join(tmECG, 'LeadV6.dat'), V6_trace-WilsonCT)

if __name__ == '__main__':
    run()    
