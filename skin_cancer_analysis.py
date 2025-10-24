import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --- 第一階段：資料檢查整理 ---
# 讀檔
data = pd.read_csv('HAM10000_metadata.csv')
print("--- 1. Basic Data Information ---")
data.info()


# 檢查 'age' 欄位的遺失值
print("\n--- 2. Check for Missing Values in 'age' Column ---")
print(f"Number of missing values in 'age': {data['age'].isnull().sum()}")

# 處理遺失值
mean_age = data['age'].mean()
data['age'].fillna(mean_age, inplace=True)
print(f"\nAfter filling with mean age {mean_age:.2f}, check again:")
print(f"Number of missing values in 'age': {data['age'].isnull().sum()}")

# 特徵工程
# 病灶類型縮寫對應到全名，建立一個包含英文完整名稱的新欄位
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
data['dx_full'] = data['dx'].map(lesion_type_dict)

# 建立一個年齡分組的新欄位，進行趨勢分析
bins = [0, 20, 40, 60, 100]
labels = ['0-20', '21-40', '41-60', '61+']
data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)

print("\n--- 已新增 'dx_full' 和 'age_group' 欄位 ---")


# --- 第二階段：單變數分析 ---

# 以下設定seaborn圖表風格
plt.style.use('seaborn-whitegrid')


# 2.1 分析病灶類型 (dx) 的分佈
plt.figure(figsize=(12, 7))
ax_dx = sns.countplot(x='dx_full', data=data, order=data['dx_full'].value_counts().index, palette='viridis')
plt.title('Distribution of Lesion Types', fontsize=16)
plt.xlabel('Lesion Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
# 在每個長條圖的頂端加上數字
for p in ax_dx.patches:
    ax_dx.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 9), textcoords='offset points')
plt.tight_layout()
plt.show()

# 2.2 分析年齡 (age) 的分佈
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution of Patients', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.tight_layout()
plt.show()


# --- 第三階段：雙變數分析---

# 3.1 根據中位數年齡排序，分析「年齡」與「病灶類型」的關係
plt.figure(figsize=(14, 8))
age_order = data.groupby('dx_full')['age'].median().sort_values().index
sns.boxplot(x='dx_full', y='age', data=data, order=age_order)
plt.title('Age Distribution for Different Lesion Types', fontsize=16)
plt.xlabel('Lesion Type', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3.2 分析「身體部位」與「病灶類型」的關係
# 建立一個交叉表
localization_dx = pd.crosstab(data['localization'], data['dx_full'])
# 繪製堆疊長條圖
localization_dx.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
plt.title('Distribution of Lesion Types across Body Parts', fontsize=16)
plt.xlabel('Body Part (Localization)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Lesion Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 3.3 分析「性別」與「病灶類型」的關係
plt.figure(figsize=(12, 7))
sns.countplot(x='dx_full', hue='sex', data=data, order=data['dx_full'].value_counts().index, palette='pastel')
plt.title('Distribution of Lesion Types by Sex', fontsize=16)
plt.xlabel('Lesion Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Sex')
plt.tight_layout()
plt.show()

# 進階insight

# 3.1 Q1: 隨著年齡增長，哪一類病灶的出現機率顯著增加？
print("\n--- 3.1 Q1: Age-related increase analysis ---")
age_dx_crosstab = pd.crosstab(data['age_group'], data['dx_full'], normalize='index') * 100
print("Percentage distribution of lesion types across age groups:")
print(age_dx_crosstab.round(2))
# 繪製百分比堆疊長條圖
age_dx_crosstab.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
plt.title('Percentage of Lesion Types across Age Groups', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Lesion Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 3.2 Q2 & Q3: 惡性腫瘤和良性痣主要分佈在哪裡？
print("\n--- 3.2 Q2 & Q3: Localization analysis ---")
malignant_lesions = data[data['dx'].isin(['mel', 'bcc'])]
benign_nevi = data[data['dx'] == 'nv']
# 繪製惡性腫瘤位置分佈
plt.figure(figsize=(10, 6))
sns.countplot(y='localization', data=malignant_lesions, order=malignant_lesions['localization'].value_counts().index, palette='Reds_r')
plt.title('Localization of Malignant Lesions (Melanoma & BCC)', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Body Part', fontsize=12)
plt.tight_layout()
plt.show()
# 繪製良性痣位置分佈
plt.figure(figsize=(10, 6))
sns.countplot(y='localization', data=benign_nevi, order=benign_nevi['localization'].value_counts().index, palette='Greens_r')
plt.title('Localization of Benign Nevi', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Body Part', fontsize=12)
plt.tight_layout()
plt.show()

# 3.3 Q4: 找出最高風險的組合特徵（以惡性黑色素瘤為例）
print("\n--- 3.3 Q4: Highest risk profile for Melanoma ---")
melanoma_data = data[data['dx'] == 'mel']
print(f"Total number of Melanoma cases: {len(melanoma_data)}")
# 分析特徵分佈
print("\nMelanoma Cases - Age Group Distribution (%):")
print(melanoma_data['age_group'].value_counts(normalize=True).round(3) * 100)
print("\nMelanoma Cases - Sex Distribution (%):")
print(melanoma_data['sex'].value_counts(normalize=True).round(3) * 100)
print("\nMelanoma Cases - Localization Distribution (%):")
print(melanoma_data['localization'].value_counts(normalize=True).round(3) * 100)
# 找出最常見的組合
risk_profile = melanoma_data.groupby(['age_group', 'sex', 'localization']).size().reset_index(name='counts')
highest_risk = risk_profile.sort_values(by='counts', ascending=False).iloc[0]
print("\n--- Highest Risk Combination for Melanoma ---")
print(f"Age Group: {highest_risk['age_group']}")
print(f"Sex: {highest_risk['sex']}")
print(f"Localization: {highest_risk['localization']}")
print(f"Number of cases in this combination: {highest_risk['counts']}")
