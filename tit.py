#%% md

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%% md

#%%
tit=sns.load_dataset('titanic')

#%%
titanic=sns.load_dataset('titanic')
#%% md

#%%
tit.head() #вивела перші 5 рядків для перевірки-датасет є
#%% md
#Перевір типи стовпців. Які з них потребують перетворення?
#%%
tit.info()
#Перетворити варто тип стовпців pclass, embarked, embark_town, who, alive, sex на 'category'.
#З огляду на обмежений перелік значень, які представлені по кожному стовпцю та варіативність значень, їх можна віднести до категорій,
#тому заміна типів саме по цих стовпцях із типу 'object' чи 'int64' на 'category' буде доречною.
#%%
tit['pclass']=tit['pclass'].astype('category')
tit['embarked']=tit['embarked'].astype('category')
tit['embark_town']=tit['embark_town'].astype('category')
tit['who']=tit['who'].astype('category')
tit['alive']=tit['alive'].astype('category')
tit['sex']=tit['sex'].astype('category')
#%%
tit.info() #Перевірка оновлених типів
#%% md

#%%
tit.describe(include='all')
#%% md

#%%
tit['survived'].count()
#%%
clean_tit=tit.drop_duplicates()
#%%
clean_tit['survived'].count()
#%%
tit['survived'].count()-clean_tit['survived'].count()

#%%
#Вилучено 107 рядків, які повністю дублюються.
#%% md

#%%
# Створюємо копію DataFrame, щоб працювати без попереджень
clean_tit = clean_tit.copy()

# Створюємо новий стовпець relatives
clean_tit['relatives'] = clean_tit['parch'] + clean_tit['sibsp']

# Видаляємо непотрібні стовпці
clean_tit = clean_tit.drop(columns=['sibsp', 'parch', 'alone'])

#%%
clean_tit.info() #Перевірка стовпців
#%% md

#%%
clean_tit['relatives'].value_counts()
#%%
clean_tit['relatives'].value_counts().plot(kind='bar')
plt.xlabel('Кількість родичів')
plt.ylabel('Кількість пасажирів')
plt.title('Розподіл пасажирів за кількістю родичів на борту')
#Найбільше пасажирів продорожує самотньо,
#далі спостерігаємо, що кількість пасажирів зменшується зі зростанням кількості родичів.
#Лише 2 пасажири подорожують із 10-ма родичами.
#%% md

#%%
clean_tit['relatives']=clean_tit['relatives'].apply(lambda x: "above 5" if x>5 else x)
#%%
clean_tit['relatives'].value_counts()#Перевірка виконання умови "above 5"
#%% md

#%%
clean_tit['relatives'] = pd.Categorical(clean_tit['relatives'], ordered=True)
#pd.Categorical автоматично поставив 'above 5' в кінець, оскільки це єдине текстове значення,
#тому явно задавати позицію в даному випадку не потрібно
#%%
clean_tit['relatives'].value_counts(sort=False)
#%%
clean_tit.info()#Стовпець 'relatives' тепер має тип 'category'
#%% md

#%%
age_median=clean_tit['age'].median() #знайшли медіану стовпця 'age' - 28.25
clean_tit['age']=clean_tit['age'].fillna(age_median)
#%%
clean_tit['age'].nunique()#Перевірка кількості унікальних значень
#%%
clean_tit['age'].value_counts()
#%% md

#%%
def age_dist(age):
    if age<14: return 'до 14 р.'
    elif 14<=age<=34: return '14-34 р.'
    elif 35<=age<=59: return '35-59 р.'
    elif age>=60: return 'старше 60 р.'
    else: return 'Unknown'
clean_tit['age_range']=clean_tit['age'].apply(age_dist)
#%%
clean_tit['age_range'].value_counts()
#%%
clean_tit.head()
#%% md

#%%
clean_tit['alive'] = titanic['alive']#проміжний рядок для відновлення оригінальних значень стовпця 'alive' з оригінального датасету
#після помилки застосування astype
#%%
clean_tit['alive'] = clean_tit['alive'] == 'yes'
#%%
clean_tit['alive'].dtype
#%%
clean_tit['alive'].value_counts()
#%%
total=clean_tit[clean_tit['age_range'] != 'Unknown'].groupby('age_range')['alive'].count()
survived=clean_tit[clean_tit['age_range'] != 'Unknown'].groupby('age_range')['alive'].sum()
#Прибрано категорію Unknown для чистоти аналізу відносного показника смертності за віковою категорією
related=np.round((((total-survived)/total)*100),2)
print(related)
#%%
#Найвища смертність спостерігається у віковій групі старше 60 р., за нею по рейтингу йде група 14-34 р., потім пасажири віком 35-49 років.
#Найнижчий % смертності спостерігаємо у групі віком до 14 р.
#%% md

#%%
all_dead=clean_tit[(clean_tit['alive'] == False) & (clean_tit['age_range'] != 'Unknown')].shape[0]
print(all_dead)

#%%
dead_in_group=clean_tit[(clean_tit['alive'] == False) & (clean_tit['age_range'] != 'Unknown')].groupby('age_range').size()
print(dead_in_group)
#%%
dead_range=np.round((dead_in_group/all_dead)*100,2)
print(dead_range)
#%%
plt.pie(
    dead_in_group / all_dead,
    labels=dead_in_group.index,
    autopct='%1.1f%%',
    startangle=90
)
plt.title('Структура загиблих за віковими групами')
plt.show()
#%%
#Найбільша частка загиблих припадає на вікову групу 13-34 р. Це може бути пов'язано з загальною кількістю пасажирів даного віку, адже саме їх
#було найбільше. Серед молодих людей мало заможніх та впливових, тобто вони могли займати 2-3 клас та дальні каюти, що також
#збільшує ризик не потрапити на рятувальний човен. Також серед них могли бути родичі членів екіпажу, які залишались із рідними.
#Найменша частка припадає на пасажирів віком старше 60 років, що також пов'язане з їх порівняно малою кількістю, а також, що ці пасажири могли займати
#перший клас, ближні каюти, що допомагало їм швидше рятуватись. З невеликим відривом від людей похилого віку йдуть діти до 14 років - за порядком їх
#рятують першими, надають пріорітет, тому частка їх смертності доволі низька.
#%%
