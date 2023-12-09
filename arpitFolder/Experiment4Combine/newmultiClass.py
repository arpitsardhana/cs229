import pandas as pd
import numpy as np
from pathlib import Path

df1 = pd.read_csv(Path("heart_failure_clinical_records_dataset.csv"))
df2 = pd.read_csv(Path("heart_statlog_cleveland_hungary_final.csv"))

class_0_count = df2[df2['target'] == 0].shape[0]
class_1_count = df2[df2['target'] == 1].shape[0]

if class_0_count > class_1_count:
    df_class_0 = df2[df2['target'] == 0].sample(class_1_count, random_state=42)
    df2 = pd.concat([df_class_0, df2[df2['target'] == 1]], ignore_index=True)
else:
    df_class_1 = df2[df2['target'] == 1].sample(class_0_count, random_state=42)
    df2 = pd.concat([df_class_1, df2[df2['target'] == 0]], ignore_index=True)

class_0_count = df1[df1['DEATH_EVENT'] == 0].shape[0]
class_1_count = df1[df1['DEATH_EVENT'] == 1].shape[0]

if class_0_count > class_1_count:
    df_class_0 = df1[df1['DEATH_EVENT'] == 0].sample(class_1_count, random_state=42)
    df1 = pd.concat([df_class_0, df1[df1['DEATH_EVENT'] == 1]], ignore_index=True)
else:
    df_class_1 = df1[df1['DEATH_EVENT'] == 1].sample(class_0_count, random_state=42)
    df1 = pd.concat([df_class_1, df1[df1['DEATH_EVENT'] == 0]], ignore_index=True)

# Create an empty DataFrame for the merged dataset
merged_columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                   'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
                   'smoking', 'time', 'DEATH_EVENT', 'chest pain type', 'resting bp s', 'cholesterol',
                   'fasting blood sugar', 'resting ecg', 'max heart rate', 'exercise angina',
                   'oldpeak', 'ST slope', 'target', 'output']

merged_df = pd.DataFrame(columns=merged_columns)

# Iterate through the rows of df1 and merge with df2 based on conditions
# Create an empty DataFrame for the merged dataset
j  = 0
print(df1.describe())
print(df2.describe())
# Iterate through rows in df1
def get_merged(merged_df, df1, df2, val1, val2, k):
    j = 0
    for index1, row1 in df1.iterrows():
        # Iterate through rows in df2
        for index2, row2 in df2.iterrows():
            # Check conditions
            if (abs(row1['age'] - row2['age']) <= 5) and row1['sex'] == row2['sex'] and row1['DEATH_EVENT'] == val1 and row2['target'] == val2:
                # Concatenate rows and calculate the average age
                age = (row1['age'] + row2['age']) // 2
                sex = row1['sex']
                nr1 = row1.drop('age')
                nr2 = row2.drop('sex')
                merged_row = pd.concat([nr1, nr2], ignore_index=False)
                merged_row['age'] = age
                merged_row['sex'] = sex

                # Add the Output column based on previous message conditions
                merged_row['output'] = int(row1['DEATH_EVENT'] + row2['target'])
                if row1['DEATH_EVENT'] == 0 and row2['target'] == 0:
                    merged_row['output'] = 0
                elif row1['DEATH_EVENT'] == 1 and row2['target'] == 1:
                    merged_row['output'] = 2
                else:
                    merged_row['output'] = 1
                #merged_row['output'] = np.where((row1['DEATH_EVENT'] == 0) & (row2['target'] == 0), 0,
                #                                np.where((row1['DEATH_EVENT'] == 1) ^ (row2['target'] == 1), 1, 2))
                #print(merged_row)
                # Append the merged row to the final dataset
                #print(row1)
                #print(row2)
                merged_df.loc[len(merged_df)] = merged_row
                #print(merged_row.to_dict())
                #merged_df = merged_df.append(merged_row, ignore_index=True)
                #merged_df = pd.concat([merged_df, merged_row], ignore_index=True)
                #print(merged_df)

                #merged_df = pd.concat([merged_df, merged_row], ignore_index=False)
                j += 1
                if j > k:
                    break
        if j > k:
            break
    return merged_df
merged_df = get_merged(merged_df, df1, df2, 0, 0, 1000)
merged_df = get_merged(merged_df, df1, df2, 1, 1, 1000)
merged_df = get_merged(merged_df, df1, df2, 0, 1, 500)
merged_df = get_merged(merged_df, df1, df2, 1, 0, 500)

# Display the merged dataset
print(merged_df.info())
#print(merged_df.describe())
filename = 'merged_data.csv'
merged_df.to_csv(filename, sep=',', index=False, encoding='utf-8')
#print(merged_df[0])

#combined_df = pd.DataFrame(np.array(np.meshgrid(df1.values, df2.values)).T.reshape(-1, 2))

#combined_df = pd.merge(df1, df2, on='sex', suffixes=('_dataset1', '_dataset2'))
#combined_df = combined_df[abs(combined_df['age_dataset1'] - combined_df['age_dataset2']) <= 5]
#combined_df['output'] = np.where((combined_df['DEATH_EVENT'] == 0) & (combined_df['target'] == 0), 0,
                                #np.where((combined_df['DEATH_EVENT'] == 1) ^ (combined_df['target'] == 1), 1, 2))

#print(combined_df.head())
