import generic_solvers as gs
import numpy as np
import dnaplotlib



class CircuitGenerator:
    '''
    This is a class that will generate models for the cyber circuits project.
    It also uses the  same formalism from dnaplotlib (link goes here) to generate
    plots, and creates generic models via the SSIT (link goes here).
    '''
    def get_repression_hill_str(self,part,rep_ind,control=False):
        '''
        get a string of a hill function with parameters determined by part and regulation.
        '''
        ux = 'x['+str(rep_ind)+']'
        return ''.join(['kb_',part['name'],'+kmax_',part['name'],'**n_',part['name'],
                                    '/(kd_',part['name']+'**n_',part['name'],'+',ux,'**n_',part['name'],')'])

    def get_activation_hill_str(self,part,rep_ind):
        '''
        get a string of a hill function with parameters determined by part and regulation.
        '''
        ux = 'x['+str(rep_ind)+']'
        return ''.join(['kb_',part['name'],'+(kmax_',part['name'],'*',ux,'**n_',part['name'],
                                    ')/(kd_',part['name']+'**n_',part['name'],'+',ux,'**n_',part['name'],')'])

    def transcription(self,part,regulation_type,species_start_ind,regulation_ind=None):
        '''
        Get the part for a single gene, transcription only.
        part:  (dictionary) from circuit part.
        '''
        stoich_matrix = np.array([[1],[-1]])
        uo = 'x['+str(species_start_ind)+']'
        if regulation_type == 'constitutive':
            propensities = ['kr_'+part['name'],'gr_'+part['name']+'*'+uo]
        elif regulation_type == 'repression':
            propensities = [self.get_repression_hill_str(part,regulation_ind),'gr_'+part['name']+'*'+uo]
        elif regulation_type == 'control':
            propensities = [0,'gr_'+part['name']+'*'+uo]
        return stoich_matrix,propensities

    def transcription_translation(self,part,regulation_type,species_start_ind,regulation_ind=None):
        '''
        Get the part for a single gene, transcription and translation.
        part:  (dictionary) from circuit part.
        '''
        uo = 'x['+str(species_start_ind)+']'
        uo1= 'x['+str(species_start_ind+1)+']'
        stoich_matrix = np.array([[1,0],[-1,0],[0,1],[0,-1]])
        if regulation_type == 'constitutive':
            propensities = ['kr_'+part['name'],'gr_'+part['name']+'*'+uo,'kp_'+part['name']+'*'+uo,'gp_'+part['name']+'*'+uo1]

        elif regulation_type == 'repression':
            propensities = [self.get_repression_hill_str(part,regulation_ind),
                                   'gr_'+part['name']+'*'+uo,
                                   'kp_'+part['name']+'*'+uo,'gp_'+part['name']+'*'+uo1]
        elif regulation_type == 'control':
            propensities = [0,
                                   'gr_'+part['name']+'*'+uo,
                                   'kp_'+part['name']+'*'+uo,'gp_'+part['name']+'*'+uo1]

        return stoich_matrix,propensities

    def bursting_transcription(self,part,regulation_type,species_start_ind,regulation_ind=None):
        '''
        Get the part for a single gene, with promoter switching, transcription only.
        part:  (dictionary) from circuit part.
        '''
        stoich_matrix = np.array([[-1,1,0],[1,-1,0],[0,0,1],[0,0,-1]])
        uo = 'x['+str(species_start_ind)+']'
        uo1 = 'x['+str(species_start_ind+1)+']'
        uo2 = 'x['+str(species_start_ind+2)+']'
        if regulation_type == 'constitutive':
            propensities = ['kon_'+part['name']+'*'+uo1,'koff_'+part['name']+'*'+uo2,'kr_'+part['name']+'*'+uo1,'gr_'+part['name']+'*'+uo2]
        elif regulation_type == 'activation':
            # increase kon by some other species
            propensities = [self.get_activation_hill_str(part,regulation_ind),
                                    'koff'+part['name'],'kr_'+part['name']+'*'+uo1,'gr_'+part['name']+'*'+uo2]
        elif regulation_type == 'repression':
            # increase kon by some other species
            propensities = ['kon_'+part['name'],
                                    self.get_activation_hill_str(part,regulation_ind),'kr_'+part['name']+'*'+uo1,'gr_'+part['name']+'*'+uo2]
        return stoich_matrix,propensities

    def parse_interactions(self,all_parts,interactions):
        '''
        parse the interactions to return the correct parts.
        '''
        part_pair_names = self.get_part_pair_names(all_parts)
        species_maps = self.get_species_maps(all_parts)
        nspec = len(species_maps)
        interaction_map = np.zeros((nspec,nspec))
        interaction_by = {}
        for i in range(len(interactions)):
            # for each interaction, find the 'to gene'
            for j in range(len(part_pair_names)):
                if interactions[i]['to_part']['name'] == part_pair_names[j][0]:
                    to_gene = part_pair_names[j][1]
                    from_gene = interactions[i]['from_part']['name']
                    # write another dumb loop to find the species for this
                    for part in part_pair_names:
                        if part[1]  == from_gene:
                            interaction_by[to_gene] = species_maps[part[2]][0]
        return interaction_by

    def get_part_pair_names(self,all_parts):
        '''
        Extract species names from an interable of all parts.
        '''
        part_pairs = []; tmp_cds = ''; tmp_cds_spec = ''; tmp_prom = ''
        for i in range(len(all_parts)):
            if (all_parts[i]['type'] == 'Promoter') or (all_parts[i]['type'] == 'CDS'):
                if all_parts[i]['type'] == 'Promoter':
                    tmp_prom = all_parts[i]['name']
                elif all_parts[i]['type'] == 'CDS':
                    # append to the name of the thing that gets appended
                    tmp_cds = all_parts[i]['name']
                    if all_parts[i]['model_type'] == 'transcription':
                        tmp_cds_spec = all_parts[i]['name']+'_rna'
                    elif all_parts[i]['model_type'] == 'transcription_translation':
                        tmp_cds_spec = all_parts[i]['name']+'_protein'
                    elif all_parts[i]['model_type'] == 'bursting_transcription':
                        tmp_cds_spec = all_parts[i]['name']+'_rna'

                part_pairs.append((tmp_prom,tmp_cds,tmp_cds_spec))
        return part_pairs[1::2]

    def get_species_maps(self,all_parts):
        '''
        Get the name and index of each species based on the type of model
        '''
        species_maps = {}
        count = 0
        for i in range(len(all_parts)):
            if all_parts[i]['type'] == 'CDS':
                if all_parts[i]['model_type'] == 'transcription':
                    # first entry is the index of the species, second is the part name.
                    species_maps[all_parts[i]['name']+'_rna']=(count,all_parts[i]['name'])
                    species_maps[all_parts[i]['name']]=count
                    count+=1
                elif all_parts[i]['model_type'] == 'transcription_translation':
                    species_maps[all_parts[i]['name']+'_rna']=(count,all_parts[i]['name'])
                    species_maps[all_parts[i]['name']+'_protein']=(count+1,all_parts[i]['name'])
                    species_maps[all_parts[i]['name']]=count
                    count += 2
                elif all_parts[i]['model_type'] == 'bursting_transcription':
                    species_maps[all_parts[i]['name']+'_g_on']=(count,all_parts[i]['name'])
                    species_maps[all_parts[i]['name']+'_g_off']=(count+1,all_parts[i]['name'])
                    species_maps[all_parts[i]['name']+'_rna']=(count+2,all_parts[i]['name'])
                    species_maps[all_parts[i]['name']]=count
                    count += 3
        self.nspec = count
        return species_maps

    def assemble_model(self,all_parts,interactions):
        '''
        Assemble the model from dictionaries.
        '''
        # first, get the species maps.
        self.species_maps = self.get_species_maps(all_parts)
        interaction_by = self.parse_interactions(all_parts,interactions)
        part_pairs = self.get_part_pair_names(all_parts)
        # Next, get the stoichiometry matrix and propensity functions
        self.propensities = []
        all_stoich_mats = []
        self.control_inds = []
        for i in range(len(interactions)):
            promoter = interactions[i]['to_part']['name']
            if interactions[i]['type'] == 'Control':
                # get the part who is being affected
                for part in part_pairs:
                    if promoter==part[0]:
                        gene = part[1]
                # get the stoichiometry and stuff for that gene
                for j in range(len(all_parts)):
                    if all_parts[j]['name'] == gene:
                        # get the index where this species starts.
                        species_start_ind = self.species_maps[gene]
                        # get the index of the species which regulates this gene
                        regulator_species_ind = 0
                        if all_parts[j]['model_type'] == 'transcription':
                            stoich,prop = self.transcription(all_parts[j],interactions[i]['type'].lower(),species_start_ind,regulator_species_ind)
                            self.control_inds.append(len(self.propensities))
                        elif all_parts[j]['model_type'] == 'transcription_translation':
                            stoich,prop = self.transcription_translation(all_parts[j],interactions[i]['type'].lower(),species_start_ind,regulator_species_ind)
                            self.control_inds.append(len(self.propensities))
                        elif all_parts[j]['model_type'] == 'bursting_transcription':
                            stoich,prop = self.bursting_transcription(all_parts[j],interactions[i]['type'].lower(),species_start_ind,regulator_species_ind)
                        self.propensities.append(prop)
                        all_stoich_mats.append(stoich)
            else:
                # get the part who is being affected
                # downstream gene
                for part in part_pairs:
                    if promoter==part[0]:
                        gene = part[1]
                # get the stoichiometry and stuff for that gene
                for j in range(len(all_parts)):
                    if all_parts[j]['name'] == gene:
                        # get the index where this species starts.
                        species_start_ind = self.species_maps[gene]
                        # get the index of the species which regulates this gene
                        regulator_species_ind = interaction_by[gene]
                        if all_parts[j]['model_type'] == 'transcription':
                            stoich,prop = self.transcription(all_parts[j],interactions[i]['type'].lower(),species_start_ind,regulator_species_ind)
                        elif all_parts[j]['model_type'] == 'transcription_translation':
                            stoich,prop = self.transcription_translation(all_parts[j],interactions[i]['type'].lower(),species_start_ind,regulator_species_ind)
                        elif all_parts[j]['model_type'] == 'bursting_transcription':
                            stoich,prop = self.bursting_transcription(all_parts[j],interactions[i]['type'].lower(),species_start_ind,regulator_species_ind)
                        self.propensities.append(prop)
                        all_stoich_mats.append(stoich)
        # make the matrix
        self.propensities = sum(self.propensities,[])
        nrxn = 0
        for k in range(len(all_stoich_mats)):
            nrxn += all_stoich_mats[k].shape[0]
        self.stoichiometry_matrix = np.zeros((nrxn,self.nspec))
        ks,ss = [0,0]
        for k in range(len(all_stoich_mats)):
            self.stoichiometry_matrix[ks:ks+all_stoich_mats[k].shape[0],ss:ss+all_stoich_mats[k].shape[1]] = all_stoich_mats[k]
            ks += all_stoich_mats[k].shape[0]
            ss += all_stoich_mats[k].shape[1]

        # use this to assemble a generic model.
        model = gs.GenericModel()
        model.init_model(np.copy(self.stoichiometry_matrix),np.copy(self.propensities))
        return model

    def get_ode_solver(self):
        '''
        Get a model object that can solve the ode version of this model.
        '''
        pass
