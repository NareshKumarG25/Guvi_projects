import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


file1= r'D:\Naresh\GUVI\Projects\Microsoft\raw_data\GUIDE_Train.csv'
file2 = r'D:\Naresh\GUVI\Projects\Microsoft\raw_data\GUIDE_Test.csv'
file3=r'D:\Naresh\GUVI\Projects\Microsoft\raw_data\new_train_sample.csv'


# train = pd.read_csv(file1)
train_2=pd.read_csv(file3)
test = pd.read_csv(file2)

selected_features = ['OrgId', 'DetectorId','AlertTitle','AlertId','Category','CountryCode','EntityType','IncidentGrade'] 

df_filtered=train_2[selected_features]
df_filtered = df_filtered[df_filtered['IncidentGrade'].notna()]

df_test = test[selected_features]
df_test=df_test[df_test['IncidentGrade'].notna()]


categorical_features=[]
numerical_features =[]
for i in df_filtered.columns:

    if type(df_filtered[i][0])== str:
        categorical_features.append(i)
    else:
        numerical_features.append(i)

le = LabelEncoder()
for i in categorical_features:
    df_filtered[i]=le.fit_transform(df_filtered[i])
    df_test[i]=le.fit_transform(df_test[i])

sample_train = df_filtered.head(500000)
sample_test = df_test.head(500000)



y_train = sample_train['IncidentGrade']
X_train = sample_train.drop('IncidentGrade', axis=1)  

y_test = sample_test['IncidentGrade']
X_test = sample_test.drop('IncidentGrade', axis=1)  


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print(f'Length of Train data : {len(sample_train)}')
print(f'Length of Test data : {len(sample_test)}')
print("------------------------------------------------")

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy on Test Data: {:.2f}%".format(accuracy * 100))

# Additional evaluation metrics
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))