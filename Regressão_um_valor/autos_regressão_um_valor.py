# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 19:06:45 2023

@author: KeichiTS
"""
import pandas as pd

base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')
base = base.drop('dateCrawled', axis = 1)
base = base.drop("dateCreated", axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)

base['name'].value_counts()
base = base.drop('name', axis = 1)
base['seller'].value_counts()
base = base.drop('seller', axis = 1)
base['offerType'].value_counts()
base = base.drop('offerType', axis = 1)

i1 = base.loc[base.price <= 10]
base.price.mean()
base = base[base.price > 10]
i2 = base.loc[base.price > 350000]
base = base.loc[base.price < 350000]

base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() #limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts()     #manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts()     #golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts()  #benzin
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts()  #nein

valores = {'vehicleType' : 'limousine', 'gearbox' : 'manuell',
           'model' : 'golf', 'fuelType' : 'benzin'
           , 'notRepairedDamage' : 'nein'}
base = base.fillna(value = valores)

previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()


previsores[:, 0] = labelencoder_previsores.fit_transform(previsores[:,0])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:, 10] = labelencoder_previsores.fit_transform(previsores[:,10])