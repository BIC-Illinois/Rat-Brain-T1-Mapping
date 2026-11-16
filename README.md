# Rat-Brain-T1-Mapping
T1 mapping for rat brains based on MDEFT and T2w MRI pre-processed by the rat branch of AIDAmri (https://github.com/Aswendt-Lab/AIDAmri/tree/rat_AIDAmri-dev-V3). This includes pre-processing (optional) and co-registration of MDEFT files to T2w images processed in the main AIDAmri workflow to obtain consistent subject-space brain masks and atlas (SIGMA, https://zenodo.org/records/10635831, https://www.nitrc.org/projects/sigma_template) labels, and T1-mapping using the curve-fit function from scipy.optimize.

## Usage
These files can be used with a native AIDAmri installation, but we recommend mounting the files to an AIDAmri container in Docker or Apptainer.

MDEFT images are saved in `sub-<subject ID>/ses-<session ID>/etc` with sidecar JSON files. Inversion Time (TI) values can be extracted from the JSON files for each MDEFT image. 

### Brain extraction for MDEFT files
These masks are not used for the T1 mapping, but pre-processing is used to support subsequent co-registration with T2w and atlas warping.

```sh
apptainer exec -B "${BASE_DIRECTORY}"/${SPECIES}/${PROJECT}:/data \
      -B ${BASE_DIRECTORY}/${SPECIES}/scripts/bet_etc.py:/aida/bin/helper_tools/bet_etc.py \
      "${IMAGEDIR}"/aidamri-"$AIDAMRI_VERSION".sif \
      python3 /aida/bin/helper_tools/bet_etc.py -i /data/proc_data_reoriented/${SUB}/${SESS}/etc/${SUB}_${SESS}_${run_id}_MDEFT.nii.gz --bet bet4animal
```

### Co-registration with T2w, atlas warping
This workflow mirrors the AIDAmri registration of other modalities to the pre-processed subject T2w image.

```sh
    apptainer exec -B "${BASE_DIRECTORY}"/${SPECIES}/${PROJECT}:/data \
      -B ${BASE_DIRECTORY}/${SPECIES}/scripts/mdeft_registration.py:/aida/bin/helper_tools/mdeft_registration.py \
      "${IMAGEDIR}"/aidamri-"$AIDAMRI_VERSION".sif \
      python3 /aida/bin/helper_tools/mdeft_registration.py -i /data/proc_data_reoriented/${SUB}/${SESS}/etc/${SUB}_${SESS}_${run_id}_MDEFT.nii.gz \
      -s /aida/lib/sigma/SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz \
      -f /aida/lib/sigma/SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz \
      -a /aida/lib/sigma/SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz\
      --sigBrain_anno /aida/lib/sigma/SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz
```

### T1-mapping
This step does not rely on software dependencies (e.g. ANTs, FSL, niftyreg) needed for AIDAmri, and can be performed in another Python environment (see requirements.txt).

```
python3 t1_mapping.py --t2w project/proc_data_reoriented/sub-001/ses-01/anat/sub-001_ses-01_run-02_T2wAntsBiasAnimalBet.nii.gz --mdeft project/proc_data_reoriented/sub-001/ses-01/etc/sub-001_ses-01_run-??_MDEFT.nii.gz --parcellation rat/hold/proc_data_reoriented/sub-001/ses-01/etc/sub-001_ses-01_run-02_MDEFT_AnnoSplit_parental.nii.gz 
```
