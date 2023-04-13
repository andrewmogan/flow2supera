import time
import edep2supera, ROOT
from ROOT import supera,std,TG4TrajectoryPoint
import numpy as np
import LarpixParser
from LarpixParser import hit_parser as HitParser
import yaml
from yaml import Loader
import larnd2supera
import objgraph
from memory_profiler import profile

class SuperaDriver(edep2supera.edep2supera.SuperaDriver):

    LOG_KEYS=['bad_track_id','packet_total','packet_ass','packet_noass','fraction_nan','ass_frac']
    def __init__(self):
        super().__init__()
        self._geom_dict  = None
        self._run_config = None
        self._trackid2idx = std.vector('supera::Index_t')()
        self._allowed_detectors = std.vector('std::string')()
        self._edeps_unassociated = []
        self._log=None
        self._electron_energy_threshold=0
        print("Initialized SuperaDriver class")


    def log(self,data_holder):

        for key in self.LOG_KEYS:
            if key in data_holder:
                raise KeyError(f'Key {key} exists in the log data holder already.')
            data_holder[key]=[]
        self._log = data_holder

    def LoadPropertyConfigs(self,cfg_dict):

        # Expect only PropertyKeyword or (TileLayout,DetectorProperties). Not both.
        if cfg_dict.get('PropertyKeyword',None):
            if cfg_dict.get('TileLayout',None) or cfg_dict.get('DetectorProperties',None):
                print('PropertyKeyword provided:', cfg_dict['PropertyKeyword'])
                print('But also founnd below:')
                for keyword in ['TileLayout','DetectorProperties']:
                    print('%s: "%s"' % (keyword,cfg_dict.get(keyword,None)))
                    print('Bool',bool(cfg_dict.get(keyword,None)))

                print('You cannot specify duplicated property infomration!')
                return False
            else:
                try:
                    self._run_config, self._geom_dict = LarpixParser.util.detector_configuration(cfg_dict['PropertyKeyword'])
                except ValueError:
                    print('Failed to load with PropertyKeyword',cfg_dict['PropertyKeyword'])
                    print('Supported types:', LarpixParser.util.configuration_keywords())
                    return False
        else:
            print('PropertyKeyword missing. Loading TileLayout and DetectorProperties...')
            if not 'TileLayout' in cfg_dict:
                print('TileLayout not in the configuration data!')
                return False

            if not 'DetectorProperties' in cfg_dict:
                print('DetectorProperties not in the configuration data!')
                raise False

            self._geom_dict  = LarpixParser.util.load_geom_dict(cfg_dict['TileLayout'])
            self._run_config = LarpixParser.util.get_run_config(cfg_dict['DetectorProperties'])

        return True


    def ConfigureFromFile(self,fname):
        with open(fname,'r') as f:
            cfg=yaml.load(f.read(),Loader=Loader)
            if not self.LoadPropertyConfigs(cfg):
                raise ValueError('Failed to configure larnd2supera!')
            self._electron_energy_threshold = cfg.get('ElectronEnergyThreshold',
                self._electron_energy_threshold
                )
        super().ConfigureFromFile(fname)


    def ConfigureFromText(self,txt):
        cfg=yaml.load(txt,Loader=Loader)
        if not self.LoadPropertyConfigs(cfg):
            raise ValueError('Failed to configure larnd2supera!')
            self._electron_energy_threshold = cfg.get('ElectronEnergyThreshold',
                self._electron_energy_threshold
                )
        super().ConfigureFromText(txt)


    @profile
    def ReadEvent(self, data, verbose=False):
        
        start_time = time.time()

        # initialize the new event record
        if not self._log is None:
            for key in self.LOG_KEYS:
                self._log[key].append(0)

        supera_event = supera.EventInput()
        supera_event.reserve(len(data.trajectories))
        print('[DRIVER] reserve', len(data.trajectories), 'for supera_event')
        
        self._trackid2idx.clear()
        self._trackid2idx.reserve(len(data.trajectories))
        self._trackid2idx.resize(len(data.trajectories), supera.kINVALID_INDEX)
        #supera_event = []

        # 1. Loop over trajectories, create one supera::ParticleInput for each
        #    store particle inputs in list to fill parent information later

        for it, traj in enumerate(data.trajectories):
        #for it in range(10):
            traj = data.trajectories[it]
            print('trajectory', it + 1, 'of', len(data.trajectories))
            part_input = supera.ParticleInput()

            #print('[DRIVER] traj count', it)

            part_input.valid = True
            part_input.part  = self.TrajectoryToParticle(traj, data.event_separator, data.first_trajectory_id)
            print('[DRIVER] supera_event.size', supera_event.size())
            part_input.part.id = supera_event.size()
            print('[DRIVER] Set ID')
            #part_input.type  = self.GetG4CreationProcess(traj)
            #supera_event.append(part_input)
            if self.GetLogger().verbose():
                if verbose:
                    print('  TrackID',part_input.part.trackid,
                          'PDG',part_input.part.pdg,
                          'Energy',part_input.part.energy_init)
            if traj['trackID'] < 0:
                print('Negative track ID found',traj['trackID'])
                raise ValueError
            #print('[DRIVER] trackid2idx initial:', self._trackid2idx)
            #print('traj[trackID]', traj['trackID'])
            #self._trackid2idx.resize(int(traj['trackID']+1),supera.kINVALID_INDEX)
            #self._trackid2idx.resize(it+1,supera.kINVALID_INDEX)

            self._trackid2idx[int(traj['trackID'])] = part_input.part.id
            print('[DRIVER] Set trackid2idx')
            #self._trackid2idx[it] = int(traj['trackID'])

            #self._trackid2idx.resize(it+1,supera.kINVALID_INDEX)
            #self._trackid2idx[it] = part_input.part.id
            #print('[DRIVER] trackid2idx:', self._trackid2idx)
            #print('[DRIVER] part_input.part.id', part_input.part.id)
            supera_event.push_back(part_input)
            print('[DRIVER] Pushed back')

        print('[DRIVER] final trackid2idx:', self._trackid2idx)
        print('[DRIVER] final len trackid2idx:', len(self._trackid2idx))
            
        if verbose:
            print("--- trajectory filling %s seconds ---" % (time.time() - start_time)) 
        start_time = time.time()  

        # 2. Fill parent information for ParticleInputs created in previous loop
        for i,part in enumerate(supera_event):
            traj = data.trajectories[i]
            print('[PARENT] traj[trackID]', traj['trackID'])

            parent=None            
            if(part.part.parent_trackid < self._trackid2idx.size()):
                print('[PARENT] part.part.parent_trackid', part.part.parent_trackid)
                #parent_index = self._trackid2idx[part.part.parent_trackid]
                parent_index = next((i for i, x in enumerate(self._trackid2idx) if x == part.part.parent_trackid), None)
                print('[PARENT] parent_index', parent_index)
                if not parent_index == supera.kINVALID_INDEX:
                    print('[PARENT] len supera_event', len(supera_event))
                    parent = supera_event[parent_index].part
                    part.part.parent_pdg = parent.pdg
                    
            self.SetProcessType(traj,part.part,parent)

        # 3. Loop over "voxels" (aka packets), get EDep from xyz and charge information,
        #    and store in pcloud
        x, y, z, dE = HitParser.hit_parser_energy(data.t0, data.packets, self._geom_dict, self._run_config)
        print('******Starting packet nonsense**********')
        print('len packets:', len(data.packets))
        if verbose:
            print('Got x,y,z,dE = ', x, y, z, dE)

        start_time = time.time()  

        track_ids = data.mc_packets_assn['track_ids']
        if verbose:
            print('track_ids before:\n', track_ids)
        fractions = data.mc_packets_assn['fraction']
        if verbose:
            print('initial_index*(track_ids!=-1):\n', data.first_track_id*(track_ids!=-1))
        
        # Check if the data.first_track_id is consistent with track_ids.
        bad_track_ids = [ v for v in np.unique(track_ids[track_ids<data.first_track_id]) if not v == -1]
        if len(bad_track_ids):
            print(f'\n[ERROR] unexpected track ID found {bad_track_ids} (first track ID in the event {data.first_track_id})')
            for bad_tid in bad_track_ids:
                print(f'    Track ID {bad_tid} ... associated fraction {fractions[np.where(track_ids==bad_tid)]}')
            print()

        if not self._log is None:
            if len(bad_track_ids):
                self._log['bad_track_id'][-1]+=1
            self._log['packet_total'][-1]+=len(data.packets)

            
        #print(len(np.unique(track_ids[track_ids>=0])),'valid track IDs... min ID',np.min(track_ids[np.where(track_ids>=0)]),
        #      'first track index:',data.first_track_id)
        #print('min track ID location:',np.where(track_ids == np.min(track_ids[np.where(track_ids>=0)]))[0])
        #print('first track ID location:',np.where(track_ids == data.first_track_id)[0])
        #print('How many valid track IDs below the first track ID?')
        #print(len(np.unique(track_ids[(track_ids>=0) & (track_ids<data.first_track_id)])))
        #print('Track ID range:',np.min(track_ids[track_ids>=0]),'=>',np.max(track_ids[track_ids>=0]))
        print('[DRIVER] pre-subtract track_ids', track_ids)
        track_ids = np.subtract(track_ids, data.first_track_id*(track_ids!=-1))
        print('[DRIVER] subtract track_ids', track_ids)
        #print('Track ID range:',np.min(track_ids[track_ids>=0]),'=>',np.max(track_ids[track_ids>=0]))
        if verbose:
            print('track_ids after:\n', track_ids)

        mm2cm = 0.1 # For converting packet x,y,z values
        for ip, packet in enumerate(data.packets):
            #if verbose:
            if True:
                print('*****************Packet', ip, '**********************')

            # If packet_type !=0 continue
            if packet['packet_type'] != 0: continue

            # print("ip",ip)
            #print('TrackID index:',ip,'/',len(track_ids))
            packet_track_ids = track_ids[ip]
            packet_fractions = fractions[ip]
            #if verbose:
            if True:
                print('packet_track_ids:', packet_track_ids)
                print('packet_fractions:', packet_fractions)

            # If only associated track IDs are -1, skip this packet.
            # 2023-02-12 commented out by Kazu: we should handle -1 which might be noise or induced current
            if len(np.unique(packet_track_ids))==1 and -1 in np.unique(packet_track_ids):
                print('[WARNING] found a packet unassociated with any tracks')
                if not self._log is None:
                    self._log['bad_adc'][-1]+=1
                edep = supera.EDep()
                edep.x,edep.y,edep.z,edep.e = x[ip]*mm2cm, y[ip]*mm2cm, z[ip]*mm2cm, dE
                edep.t    = packet_track['t'] 
                edep.dedx = packet_track['dEdx']
                self._edeps_unassociated.append(edep)
                if not self._log is None:
                    self._log['packet_noass'][-1] += 1
                continue

            # If fraction includes `nan` skip this packet.
            if np.isnan(packet_fractions).sum():
                print(f'    WARNING: found a packet containing {np.isnan(packet_fractions).sum()} nan fractions')
                if self._log is not None:
                    self._log['fraction_nan'][-1] += 1
                continue

            for t in range(len(packet_track_ids)):
                #if verbose:
                if True:
                    print('----Packet trackID', t, '-----')

                # 2023-02-12 commented out by Kazu: we should handle -1 which might be noise or induced current
                if packet_track_ids[t] <= -1: 
                    print('Negative packet trackID')
                    continue
                
                #print(packet_track_ids[t])
                packet_track = data.tracks[packet_track_ids[t]]
                print('packet_track_ids[t]', packet_track_ids[t])
                edep = supera.EDep()

                edep.x,edep.y,edep.z = x[ip]*mm2cm, y[ip]*mm2cm, z[ip]*mm2cm
                if verbose:
                    print('edep xyz:', edep.x, edep.y, edep.z)
                    print('Multiply dE {} by fraction {}'.format(packet_track['dE'], packet_fractions[t]))
                #edep.e    = packet_track['dE'] * packet_fractions[t]
                edep.e    = dE[ip] * packet_fractions[t]
                edep.t    = packet_track['t'] 
                edep.dedx = packet_track['dEdx']

                # register EDep
                print('[DRIVER] len supera_event', len(supera_event))
                if edep.t >= 0:
                    packet_index = next((i for i, x in enumerate(self._trackid2idx) if x == packet_track['trackID']), None)
                    #packet_index = int([i for i, x in enumerate(self._trackid2idx) if x == packet_track['trackID']])
                    print('packet_index', packet_index)
                    print('packetid2idx[packet_index]', self._trackid2idx[packet_index])
                    #supera_event[self._trackid2idx[int(packet_index)]].pcloud.push_back(edep)
                    supera_event[packet_index].pcloud.push_back(edep)
                    #supera_event[self._trackid2idx[int(packet_track['trackID'])]].pcloud.push_back(edep)

                if not self._log is None:
                    self._log['ass_frac'][-1] += packet_fractions[t]

            if not self._log is None:
                self._log['packet_ass'][-1]+=1
        

        if verbose:
            print("--- filling edep %s seconds ---" % (time.time() - start_time)) 
        if not self._log is None:
            if self._log['packet_ass'][-1]>0:
                self._log['ass_frac'][-1] /= self._log['packet_ass'][-1]

            if self._log['packet_noass'][-1]:
                value_bad, value_frac = self._log['packet_noass'][-1], self._log['packet_noass'][-1]/self._log['packet_total'][-1]
                print(f'    WARNING: {value_bad} packets ({value_frac} %) had no MC track association')
            if self._log['fraction_nan'][-1]:
                value_bad, value_frac = self._log['fraction_nan'][-1], self._log['fraction_nan'][-1]/self._log['packet_total'][-1]
                print(f'    WARNING: {value_bad} packets ({value_frac} %) had nan fractions associated')
            if self._log['ass_frac'][-1]<0.9999:
                print(f'    WARNING associated packet count fraction is {self._log["ass_frac"][-1]} (<1.0)')
            if self._log['bad_track_id'][-1]:
                print(f'    WARNING: {self._log["bad_track_id"][-1]} invalid track IDs found in the association')

        return supera_event

    #@profile
    def TrajectoryToParticle(self, trajectory, event_separator, first_trajectory_id):
        mm2cm = 0.1

        p = supera.Particle()
        # Larnd-sim stores a lot of these fields as numpy.uint32, 
        # but Supera/LArCV want a regular int, hence the type casting
        # TODO Is there a cleaner way to handle this?
        p.id             = int(trajectory[event_separator])
        #p.interaction_id = trajectory['interactionID']
        p.trackid        = int(trajectory['trackID'])
        p.pdg            = int(trajectory['pdgId'])
        p.px = trajectory['pxyz_start'][0] 
        p.py = trajectory['pxyz_start'][1] 
        p.pz = trajectory['pxyz_start'][2]
        p.energy_init = np.sqrt(pow(larnd2supera.pdg2mass.pdg2mass(p.pdg),2)+pow(p.px,2)+pow(p.py,2)+pow(p.pz,2))
        p.vtx    = supera.Vertex(trajectory['xyz_start'][0]*mm2cm, 
                                 trajectory['xyz_start'][1]*mm2cm, 
                                 trajectory['xyz_start'][2]*mm2cm, 
                                 trajectory['t_start']
        )
        p.end_pt = supera.Vertex(trajectory['xyz_end'][0]*mm2cm, 
                                 trajectory['xyz_end'][1]*mm2cm,
                                 trajectory['xyz_end'][2]*mm2cm, 
                                 trajectory['t_end']
        )

        traj_parent_id = trajectory['parentID']
        print('[TRAJ2PART] traj parent ID:', traj_parent_id)
        print('[TRAJ2PART] particle track ID:', p.trackid)
        # This now causes errors?
        #if traj_parent_id == -1: p.parent_trackid = supera.kINVALID_TRACKID
        # Subtract first_trajectory_id so that each parent index is within the range of the event trajectories
        if traj_parent_id == -1: p.parent_trackid = int(p.trackid - first_trajectory_id)
        else:                    
            print('[ELSE] trajectory[parentID]', trajectory['parentID'])
            print('[ELSE] diff:', trajectory['parentID'] - first_trajectory_id)
            p.parent_trackid = int(trajectory['parentID'] - first_trajectory_id)

        print('[TRAJ2PART] particle parent ID:', p.parent_trackid)

        if supera.kINVALID_TRACKID in [p.trackid, p.parent_trackid]:
            print('Unexpected to have an invalid track ID',p.trackid,
                  'or parent track ID',p.parent_trackid)
            raise ValueError
        
        print('[TRAJ2PART] Returning')
        return p
        
        
    def SetProcessType(self, edepsim_part, supera_part, supera_parent):
        pdg_code    = supera_part.pdg
        g4type_main = edepsim_part['start_process']
        g4type_sub  = edepsim_part['start_subprocess']
        
        supera_part.process = '%d::%d' % (int(g4type_main),int(g4type_sub))

        ke = np.sqrt(pow(supera_part.px,2)+pow(supera_part.py,2)+pow(supera_part.pz,2))

        dr = 1.e20
        if supera_parent:
            dx = (supera_parent.end_pt.pos.x - supera_part.vtx.pos.x)
            dy = (supera_parent.end_pt.pos.y - supera_part.vtx.pos.y)
            dz = (supera_parent.end_pt.pos.z - supera_part.vtx.pos.z)
            dr = dx + dy + dz
        
        if pdg_code == 2112 or pdg_code > 1000000000:
            supera_part.type = supera.kNeutron
        
        elif supera_part.trackid == supera_part.parent_trackid:
            supera_part.type = supera.kPrimary
            
        elif pdg_code == 22:
            supera_part.type = supera.kPhoton
        
        elif abs(pdg_code) == 11:
            
            if g4type_main == TG4TrajectoryPoint.G4ProcessType.kProcessElectromagetic:
                
                if g4type_sub == TG4TrajectoryPoint.G4ProcessSubtype.kSubtypeEMPhotoelectric:
                    supera_part.type = supera.kPhotoElectron
                    
                elif g4type_sub == TG4TrajectoryPoint.G4ProcessSubtype.kSubtypeEMComptonScattering:
                    supera_part.type = supera.kCompton
                
                elif g4type_sub == TG4TrajectoryPoint.G4ProcessSubtype.kSubtypeEMGammaConversion or \
                     g4type_sub == TG4TrajectoryPoint.G4ProcessSubtype.kSubtypeEMPairProdByCharged:
                    supera_part.type = supera.kConversion
                    
                elif g4type_sub == TG4TrajectoryPoint.G4ProcessSubtype.kSubtypeEMIonization:
                    
                    if abs(supera_part.parent_pdg) == 11:
                        supera_part.type = supera.kIonization
                        
                    elif abs(supera_part.parent_pdg) == 211 or abs(supera_part.parent_pdg) == 13 or abs(supera_part.parent_pdg) == 2212:
                        supera_part.type = supera.kDelta

                    elif supera_part.parent_pdg == 22:
                        supera_part.type = supera.kCompton

                    else:
                        print("    WARNING: UNEXPECTED CASE for IONIZATION ")
                        print("      PDG",pdg_code,
                              "TrackId",edepsim_part['trackID'],
                              "Kinetic Energy",ke,
                              "Parent PDG",supera_part.parent_pdg ,
                              "Parent TrackId",edepsim_part['parentID'],
                              "G4ProcessType",g4type_main ,
                              "SubProcessType",g4type_sub)
                        supera_part.type = supera.kIonization
                #elif g4type_sub == 151:

                else:
                    print("    WARNING: UNEXPECTED EM SubType ")
                    print("      PDG",pdg_code,
                          "TrackId",edepsim_part['trackID'],
                          "Kinetic Energy",ke,
                          "Parent PDG",supera_part.parent_pdg ,
                          "Parent TrackId",edepsim_part['parentID'],
                          "G4ProcessType",g4type_main ,
                          "SubProcessType",g4type_sub)
                    raise ValueError
                    
            elif g4type_main == TG4TrajectoryPoint.G4ProcessType.kProcessDecay:
                #print("    WARNING: DECAY ")
                #print("      PDG",pdg_code,
                #      "TrackId",edepsim_part['trackID'],
                #      "Kinetic Energy",ke,
                #      "Parent PDG",supera_part.parent_pdg ,
                #      "Parent TrackId",edepsim_part['parentID'],
                #      "G4ProcessType",g4type_main ,
                #      "SubProcessType",g4type_sub)
                supera_part.type = supera.kDecay

            elif g4type_main == TG4TrajectoryPoint.G4ProcessType.kProcessHadronic and g4type_sub == 151 and dr<0.0001:
                if ke < self._electron_energy_threshold:
                    supera_part.type = supera.kIonization
                else:
                    supera_part.type = supera.kDecay
            
            else:
                print("    WARNING: Guessing the shower type as", "Compton" if ke < self._electron_energy_threshold else "OtherShower")
                print("      PDG",pdg_code,
                      "TrackId",edepsim_part['trackID'],
                      "Kinetic Energy",ke,
                      "Parent PDG",supera_part.parent_pdg ,
                      "Parent TrackId",edepsim_part['parentID'],
                      "G4ProcessType",g4type_main ,
                      "SubProcessType",g4type_sub)

                if ke < self._electron_energy_threshold:
                    supera_part.type = supera.kCompton
                else:
                    supera_part.type = supera.kOtherShower
        else:
            supera_part.type = supera.kTrack
