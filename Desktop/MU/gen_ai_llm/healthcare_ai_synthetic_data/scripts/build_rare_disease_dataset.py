import pandas as pd
import numpy as np

print("=" * 60)
print("STEP 1: BUILDING RARE DISEASE EHR DATASET")
print("=" * 60)

DATA_PATH = 'synthea_sample_data_csv/csv/'

print("\n[1/5] Loading SYNTHEA tables...")
patients = pd.read_csv(DATA_PATH + 'patients.csv')
conditions = pd.read_csv(DATA_PATH + 'conditions.csv')
medications = pd.read_csv(DATA_PATH + 'medications.csv')
encounters = pd.read_csv(DATA_PATH + 'encounters.csv')
observations = pd.read_csv(DATA_PATH + 'observations.csv')

print(f"  Patients: {patients.shape}")
print(f"  Conditions: {conditions.shape}")
print(f"  Medications: {medications.shape}")
print(f"  Encounters: {encounters.shape}")
print(f"  Observations: {observations.shape}")

print("\n[2/5] Extracting patient demographics...")
patients_clean = patients[['Id', 'BIRTHDATE', 'GENDER', 'RACE', 'ETHNICITY']].copy()
patients_clean.columns = ['PATIENT', 'BIRTHDATE', 'GENDER', 'RACE', 'ETHNICITY']
patients_clean['AGE'] = (pd.to_datetime('2026-01-01') - pd.to_datetime(patients_clean['BIRTHDATE'])).dt.days / 365.25
patients_clean['AGE'] = patients_clean['AGE'].round(0).astype(int)
patients_clean = patients_clean.drop('BIRTHDATE', axis=1)
print(f"  Patients: {len(patients_clean)}")

print("\n[3/5] Building condition features...")
condition_counts = conditions.groupby('PATIENT')['DESCRIPTION'].count().reset_index()
condition_counts.columns = ['PATIENT', 'NUM_CONDITIONS']

primary_dx = conditions.groupby('PATIENT')['DESCRIPTION'].agg(lambda x: x.value_counts().index[0]).reset_index()
primary_dx.columns = ['PATIENT', 'PRIMARY_DIAGNOSIS']

rare_conditions = conditions['DESCRIPTION'].value_counts()
rare_conditions = rare_conditions[rare_conditions < 10].index.tolist()
conditions['IS_RARE'] = conditions['DESCRIPTION'].isin(rare_conditions).astype(int)
rare_flag = conditions.groupby('PATIENT')['IS_RARE'].max().reset_index()
rare_flag.columns = ['PATIENT', 'HAS_RARE_DISEASE']
print(f"  Rare conditions: {len(rare_conditions)}")
print(f"  Rare disease patients: {rare_flag['HAS_RARE_DISEASE'].sum()}")

print("\n[4/5] Building medication and encounter features...")
med_counts = medications.groupby('PATIENT')['DESCRIPTION'].count().reset_index()
med_counts.columns = ['PATIENT', 'NUM_MEDICATIONS']

enc_counts = encounters.groupby('PATIENT')['ENCOUNTERCLASS'].count().reset_index()
enc_counts.columns = ['PATIENT', 'NUM_ENCOUNTERS']

enc_types = encounters.groupby('PATIENT')['ENCOUNTERCLASS'].agg(lambda x: x.value_counts().index[0]).reset_index()
enc_types.columns = ['PATIENT', 'PRIMARY_ENCOUNTER_TYPE']

obs_numeric = observations[observations['TYPE'] == 'numeric'].copy()
if len(obs_numeric) > 0:
    obs_numeric['VALUE'] = pd.to_numeric(obs_numeric['VALUE'], errors='coerce')
    obs_means = obs_numeric.groupby('PATIENT')['VALUE'].agg(['mean', 'std', 'min', 'max']).reset_index()
    obs_means.columns = ['PATIENT', 'OBS_MEAN', 'OBS_STD', 'OBS_MIN', 'OBS_MAX']
    obs_count = obs_numeric.groupby('PATIENT')['VALUE'].count().reset_index()
    obs_count.columns = ['PATIENT', 'NUM_OBSERVATIONS']
else:
    obs_means = pd.DataFrame(columns=['PATIENT', 'OBS_MEAN', 'OBS_STD', 'OBS_MIN', 'OBS_MAX'])
    obs_count = pd.DataFrame(columns=['PATIENT', 'NUM_OBSERVATIONS'])

print("\n[5/5] Merging all features...")
merged = patients_clean.copy()
merged = merged.merge(condition_counts, on='PATIENT', how='left')
merged = merged.merge(primary_dx, on='PATIENT', how='left')
merged = merged.merge(rare_flag, on='PATIENT', how='left')
merged = merged.merge(med_counts, on='PATIENT', how='left')
merged = merged.merge(enc_counts, on='PATIENT', how='left')
merged = merged.merge(enc_types, on='PATIENT', how='left')
if len(obs_means) > 0:
    merged = merged.merge(obs_means, on='PATIENT', how='left')
    merged = merged.merge(obs_count, on='PATIENT', how='left')

merged = merged.fillna(0)
merged = merged.drop('PATIENT', axis=1)
merged.to_csv('data/raw/rare_disease_ehr.csv', index=False)

print(f"\n✓ Shape: {merged.shape}")
print(f"✓ Columns: {merged.columns.tolist()}")
print(f"✓ Rare disease patients: {int(merged['HAS_RARE_DISEASE'].sum())} / {len(merged)}")
print(f"✓ Saved to data/raw/rare_disease_ehr.csv")
print("=" * 60)