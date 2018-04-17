import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('NCSOS Pyth Regression3.csv')

this_year = df[df['Year'] == 0]
df = df[df['Year'] >= 2011]
history = df[df['Year'] != 0]
# add_to = history[history['Year'] == 2018]
# inputs = history[history['Year'] != 2018]

history_X = np.array(history[['Round', 'Seed Adv.', 'AdjEM Adv.', 'Tempo Diff.', 'NCSOS Adv.', 'KenPom', 'FiveThirtyEight']])
history_y = np.array(history['Survive'])


X_train, X_test, y_train, y_test = train_test_split(history_X, history_y, random_state=67)
clf = RandomForestClassifier(random_state=25, n_estimators=1000)
# added_X = np.array(add_to[['Seed Adv.', 'AdjEM Adv.', 'NCSOS Adv.', 'KP FanMatch']])
# added_y = np.array(add_to['Survive'])
#12:81 (52/85)  85/44 (52:18), 85/45 (67:25), 85/46 (67:21)
# X_train = np.append(X_train, added_X)
# X_train = X_train.reshape(X_train.shape[0]/4, 4)
# y_train = np.append(y_train, added_y)

this_year_X = np.array(this_year[['Round', 'Seed Adv.', 'AdjEM Adv.', 'Tempo Diff.', 'NCSOS Adv.', 'KenPom', 'FiveThirtyEight']])

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))

lg = LogisticRegression()
lg.fit(X_train,y_train)
lg_score = lg.score(X_test, y_test)
lg.predict(X_test)
print(lg_score)
