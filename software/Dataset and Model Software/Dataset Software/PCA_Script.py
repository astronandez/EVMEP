import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r'C:\Users\Marc Hernandez\Documents\UCLA\ECE 202A\EVMEP\data\custom_vehicle_data.csv')
vehicle_types = df['body_style']
df['body_style'] = pd.Categorical(df['body_style'])
df['body_style_code'] = df['body_style'].cat.codes

unique_body_style = df['body_style'].unique()
for i in unique_body_style:
    print(i)
# numeric_vehicle_info = df.drop(['body_style'], axis=1)
numeric_vehicle_info = df.drop(['body_style', "cargo_capacity_in", "horsepower_rpm", "torque_rpm", "towing_capacity_lbs","payload_capacity_lbs"], axis=1)
numeric_vehicle_info = numeric_vehicle_info.dropna(thresh=(0.9 * len(numeric_vehicle_info.columns)))

imputer = SimpleImputer(strategy='mean')
numeric_vehicle_info_imputed = pd.DataFrame(imputer.fit_transform(numeric_vehicle_info),
                                            columns=numeric_vehicle_info.columns)

scaler = StandardScaler()
final_data_prepro = scaler.fit_transform(numeric_vehicle_info_imputed)
scaled = pd.DataFrame(final_data_prepro, columns=numeric_vehicle_info_imputed.columns)

pca = PCA()
pca.fit_transform(scaled)
final_components = pd.DataFrame(pca.components_).transpose()
final_components.columns = [f"Comp {index}" for index in range(1, len(scaled.columns) + 1)]
final_components.index = scaled.columns.values

print(final_components)
explained_variance_ratio = pca.explained_variance_ratio_

desired_explained_variance = 0.9
cumulative_explained_variance = explained_variance_ratio.cumsum()
n_components = sum(cumulative_explained_variance < desired_explained_variance) + 1
print(f"\nNumber of components to retain: {n_components}")

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.title('Explained Variance Ratio = 0.9')
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance Ratio')
plt.show()

var_ratios = pd.DataFrame(explained_variance_ratio).transpose()
var_ratios.columns = [f"Comp {index}" for index in range(1, len(scaled.columns) + 1)]
var_ratios.index = ["Variance Per PC"]
print(var_ratios)

# pca2 = PCA(n_components)
pca2 = PCA()
limited_pcs = pd.DataFrame(pca2.fit_transform(scaled))
comps = pca2.components_
limited_pcs = limited_pcs.iloc[:, :n_components]
limited_pcs.columns = [f"Comp {index}" for index in range(1, len(limited_pcs.columns) + 1)]
plt.figure(figsize=(8, 5))
plt.scatter(comps[:, 0], comps[:, 1], alpha=0.5)
for i, vector in enumerate(pca2.components_):
    plt.arrow(0, 0, vector[0], vector[1], color='r', alpha=0.5)
    plt.text(vector[0], vector[1], f'Var{i+1}', color='g', ha='center', va='center')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Biplot')
plt.show()

print(limited_pcs.head())
plt.figure(figsize=(10, 5))
sns.heatmap(pca2.components_,
            cmap='Blues',
            yticklabels=[f'PC{i+1}' for i in range(pca2.n_components_)],
            xticklabels=[f'{i}' for i in scaled.columns.values],
            annot=True)
plt.title('Heatmap of PCA Component Loadings')
plt.show()



