import h5py as h5
import numpy as np
from LarpixParser import event_parser as EventParser
from LarpixParser.util import detector_configuration

class InputEvent:
    event_id = -1
    mc_packets_assn = None
    tracks  = None
    packets = None
    trajectories = None
    t0 = -1
    first_track_id = -1
    first_trajectory_id = -1
    event_separator = ''

class InputReader:
    
    def __init__(self,input_files=None,event_separator='eventID'):
        self._mc_packets_assn = None
        self._packets = None
        self._tracks = None
        self._trajectories = None
        self._vertices = None
        self._packet2event = None
        self._event_ids = None
        self._event_t0s = None
        self._if_spill = False

        if event_separator == "spillID":
            self._if_spill = True
        
        if input_files:
            self.ReadFile(input_files)

    def __len__(self):
        if self._event_ids is None: return 0
        return len(self._event_ids)


    def __iter__(self):
        for entry in range(len(self)):
            yield self.GetEntry(entry)


    def _correct_t0s(self,event_t0s,num_event):
        # compute dt.
        dt=event_t0s[1:]-event_t0s[:-1]
        print(f'    Found {(dt==0).sum()} duplicate T0 values (removing)' )
        print(f'    Entries removed: {np.where(dt==0)[0]+1}')
        # generate a mask for dt>0
        mask=np.insert(np.where(dt>0)[0]+1,0,0)
        # apply mask
        corrected_t0s = event_t0s[mask]
        return corrected_t0s

    
    def ReadFile(self,input_files,verbose=False):
        #mc_packets_assn = []
        #packets = []
        #tracks  = []
        #trajectories = []
        #vertices = []
        
        if type(input_files) == str:
            input_files = [input_files]

        self.GetDatasets(input_files,verbose)
            
        #for f in input_files:
        #    with h5.File(f,'r') as fin:
        #        mc_packets_assn.append(fin['mc_packets_assn'][:])
        #        packets.append(fin['packets'][:])
        #        tracks.append(fin['tracks'][:])
        #        trajectories.append(fin['trajectories'][:])
        #        if verbose: print('Read-in:',f)
                
        #self._mc_packets_assn = np.concatenate(mc_packets_assn)
        #self._packets = np.concatenate(packets)
        #self._tracks  = np.concatenate(tracks )
        #self._trajectories = np.concatenate(trajectories)

        # create mapping
        self._packet2event = EventParser.packet_to_eventid(self._mc_packets_assn, self._tracks,ifspill=self._if_spill)
        print('self._packet2event:', self._packet2event)
        print('len self._packet2event:', len(self._packet2event))
        
        #valid_packet_mask = self._packet2event != -1
        valid_packet_mask = self._packet2event != -1
        ctr_packet  = len(self._packets)
        ctr_invalid_packet = ctr_packet - valid_packet_mask.sum()
        if verbose:
            print('    %d (%.2f%%) packets without an event ID assignment. They will be ignored.' % (ctr_invalid_packet,
                                                                                                     ctr_invalid_packet/ctr_packet)
                 )
        print('packet2event w/ mask', self._packet2event[valid_packet_mask])
        print('len packet2event w/ mask', len(self._packet2event[valid_packet_mask]))
        # create a list of unique Event IDs
        self._event_ids = np.unique(self._packet2event[valid_packet_mask]).astype(np.int64)
        print('[READER] event_ids', self._event_ids)
        print('[READER] len event_ids', len(self._event_ids))
        if verbose:
            missing_ids = [i for i in np.arange(np.min(self._event_ids),np.max(self._event_ids)+1,1) if not i in self._event_ids]
            print('    %d unique event IDs found.' % len(self._event_ids))
            print('    Potentially missing %d event IDs %s' % (len(missing_ids),str(missing_ids)))
        
        # create a list of corresponding T0s
        #if self._if_spill:
        t0_group = np.empty(shape=(0,))
        if self._if_spill:
            # TODO Hard coding alert!
            detector = '2x2' 
            run_config,_ = detector_configuration(detector)
            t0_group = EventParser.get_t0_spill(self._vertices,run_config)
            # Match true t0s with reconstructed events
            t0_group = [t0_group[i] for i in self._event_ids]
        else:
            print('WOOOOOOOOOOOO')
            t0_group = EventParser.get_t0(self._packets)
        print('t0_group', t0_group)
        print('type t0_group', type(t0_group))
        print('len t0_group', len(t0_group))

        # Assert strong assumptions here
        # Assumption 1: currently we assume T0 is same for all readout groups
        #self._event_t0s = self._correct_t0s(np.unique(t0_group,axis=1),len(self._event_ids))
        #self._event_t0s = self._correct_t0s(np.unique(t0_group),len(self._event_ids))
        self._event_t0s = np.unique(t0_group)
        #if not self._event_t0s.shape[1]==1:
        #    print('    ERROR: found an event with non-unique T0s across readout groups.')
        #    raise ValueError('Current code implementation assumes unique T0 per event!')

        # Assumption 2: the number of readout should be same as the number of valid Event IDs
        if not len(self._event_ids) == self._event_t0s.shape[0]:
            raise ValueError('Mismatch in the number of unique Event IDs and event T0 counts')

        # Now it's safe to assume all readout groups for every event shares the same T0
        self._event_t0s = self._event_t0s.flatten()

    def GetDatasets(self,input_files,verbose):

        mc_packets_assn = []
        packets = []
        tracks  = []
        trajectories = []
        vertices = []

        for f in input_files:
            with h5.File(f,'r') as fin:
                mc_packets_assn.append(fin['mc_packets_assn'][:])
                packets.append(fin['packets'][:])
                tracks.append(fin['tracks'][:])
                trajectories.append(fin['trajectories'][:])
                vertices.append(fin['vertices'][:])
                if verbose: print('Read-in:',f)

        self._mc_packets_assn = np.concatenate(mc_packets_assn)
        self._packets = np.concatenate(packets)
        self._tracks  = np.concatenate(tracks )
        self._trajectories = np.concatenate(trajectories)
        self._vertices = np.concatenate(vertices)

    def GetEvent(self,event_id):
        
        index_loc = (self._event_ids == event_id).nonzero()[0]
        
        if len(index_loc) < 1:
            print('Event ID',event_id,'not found in the data')
            print('Invalid read request (returning None)')
            return None
        
        return GetEntry(index_loc[0])
        
    def GetEntry(self,index,event_separator):
        
        if index >= len(self._event_ids):
            print('Entry',index,'is above allowed entry index (<%d)' % len(self._event_ids))
            print('Invalid read request (returning None)')
            return None

        valid_separators = ['eventID', 'spillID']
        if event_separator not in valid_separators:
            raise ValueError(f"Event separator must be one of {valid_separators}")

        
        # Now return event info for the found index
        result = InputEvent()

        result.event_separator = event_separator
        
        result.event_id = self._event_ids[index]
        result.t0 = self._event_t0s[index]
        print('[GETENTRY] result.event_id', self._event_ids[index])

        mask = self._packet2event == result.event_id
        
        result.packets = self._packets[mask]
        result.mc_packets_assn = self._mc_packets_assn[mask]
        
        mask = self._tracks[event_separator] == result.event_id
        result.tracks = self._tracks[mask]
        #print('[GETENTRY] result.tracks:', result.tracks)
        print('[GETENTRY] len result.tracks:', len(result.tracks))
        
        result.first_track_id = mask.nonzero()[0][0]
        print('[GETENTRY] first track ID:', result.first_track_id)
        
        mask = self._trajectories[event_separator] == result.event_id
        result.trajectories = self._trajectories[mask]
        result.first_trajectory_id = mask.nonzero()[0][0]
        print('[GETENTRY] first trajectory ID:', result.first_trajectory_id)

        print('[GETENTRY] got', len(result.trajectories), 'event trajectories')
        
        return result  
