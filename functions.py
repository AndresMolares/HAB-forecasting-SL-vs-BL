import pandas as pd
import numpy as np
import sys
import random
import tensorflow as tf
from river import stream
from river import metrics
from river import preprocessing
from river import feature_selection
from river import stats
from river import tree
from river import ensemble
from river import neighbors
from river import neural_net as nn
from river import optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

sys.setrecursionlimit(9999)

def obtenerDataset(file_ui, file_sta_croco, file_sect_croco, dias_prediccion, dias_caracteristicas, zona, zonas_aux):
	lista_caracteristicas_v1 = []
	lista_caracteristicas_v2 = []

	df_DailyUI=pd.read_csv(file_ui,sep=',')
	ui = ['UI']

	df_CROCO_Dacumi=pd.read_csv(file_sta_croco,sep=',')
	month = ['Month']

	v2 =['TSur_mean'+zona, 'TSur_std'+zona, 'TBot_mean'+zona, 'TBot_std'+zona, 'SSur_mean'+zona, 'SSur_std'+zona, 'SBot_mean'+zona, 'SBot_std'+zona, 'USur_mean'+zona, 'USur_std'+zona, 'UBot_mean'+zona, 'UBot_std'+zona, 'VSur_mean'+zona, 'VSur_std'+zona, 'VBot_mean'+zona, 'VBot_std'+zona, 'BVmax'+zona, 'dept_BVmax'+zona, 'Chl'+zona,'Dacuminata'+zona]
	lista_caracteristicas_v1.append(v2)

	df_CROCO=pd.read_csv(file_sect_croco,sep=',')

	for zona_aux in zonas_aux:
		if zona_aux[0] != 'S':
			datos_zona = ['TSur_mean' + zona_aux, 'TSur_std' + zona_aux, 'TBot_mean' + zona_aux, 'TBot_std' + zona_aux,
						  'SSur_mean' + zona_aux, 'SSur_std' + zona_aux, 'SBot_mean' + zona_aux, 'SBot_std' + zona_aux,
						  'USur_mean' + zona_aux, 'USur_std' + zona_aux, 'UBot_mean' + zona_aux, 'UBot_std' + zona_aux,
						  'VSur_mean' + zona_aux, 'VSur_std' + zona_aux, 'VBot_mean' + zona_aux, 'VBot_std' + zona_aux,
						  'BVmax' + zona_aux, 'dept_BVmax' + zona_aux, 'Chl' + zona_aux, 'Dacuminata' + zona_aux]
			lista_caracteristicas_v1.append(datos_zona)
		else:
			datos_zona = ['TSur1_mean' + zona_aux, 'TSur1_std' + zona_aux, 'TBot1_mean' + zona_aux,
						  'TBot1_std' + zona_aux,
						  'SSur1_mean' + zona_aux, 'SSur1_std' + zona_aux, 'SBot1_mean' + zona_aux,
						  'SBot1_std' + zona_aux,
						  'USur1_mean' + zona_aux, 'USur1_std' + zona_aux, 'UBot1_mean' + zona_aux,
						  'UBot1_std' + zona_aux,
						  'VSur1_mean' + zona_aux, 'VSur1_std' + zona_aux, 'VBot1_mean' + zona_aux,
						  'VBot1_std' + zona_aux,
						  'TSur2_mean' + zona_aux, 'TSur2_std' + zona_aux, 'TBot2_mean' + zona_aux,
						  'TBot2_std' + zona_aux,
						  'SSur2_mean' + zona_aux, 'SSur2_std' + zona_aux, 'SBot2_mean' + zona_aux,
						  'SBot2_std' + zona_aux,
						  'USur2_mean' + zona_aux, 'USur2_std' + zona_aux, 'UBot2_mean' + zona_aux,
						  'UBot2_std' + zona_aux,
						  'VSur2_mean' + zona_aux, 'VSur2_std' + zona_aux, 'VBot2_mean' + zona_aux,
						  'VBot2_std' + zona_aux]
			lista_caracteristicas_v2.append(datos_zona)

	dataset=pd.DataFrame(df_CROCO_Dacumi[month].iloc[0:df_CROCO_Dacumi.shape[0], :])
	dataset=dataset.shift(periods=((dias_caracteristicas-1) * -1))

	for i in range(dias_caracteristicas):
		dataset = pd.concat(
			[dataset, df_DailyUI[ui].iloc[0+i:df_CROCO_Dacumi.shape[0] - (dias_prediccion-i), :].reset_index(drop=True)],
			axis=1)

	for bloques_caracteristicas in lista_caracteristicas_v1:
		for i in range(dias_caracteristicas):
			dataset=pd.concat([dataset, df_CROCO_Dacumi[bloques_caracteristicas].iloc[0+i:df_CROCO_Dacumi.shape[0]-(dias_prediccion-i),:].reset_index(drop=True)], axis=1)
	for bloques_caracteristicas in lista_caracteristicas_v2:
		for i in range(dias_caracteristicas):
			dataset=pd.concat([dataset, df_CROCO[bloques_caracteristicas].iloc[0+i:df_CROCO_Dacumi.shape[0]-(dias_prediccion-i),:].reset_index(drop=True)], axis=1)

	return dataset, lista_caracteristicas_v1, lista_caracteristicas_v2, df_CROCO_Dacumi

def reetiquetarDataset(dataset, lista_caracteristicas_v1, lista_caracteristicas_v2, df_CROCO_Dacumi, zona, dias_prediccion, dias_caracteristicas):
	columnas = ['Month']

	for i in range(dias_caracteristicas):
		columnas.append('UI_' + str(dias_caracteristicas-1-i))

	for bloques_caracteristicas in lista_caracteristicas_v1:
		for i in range(dias_caracteristicas):
			for car in bloques_caracteristicas:
				columnas.append(car + '_' + str(dias_caracteristicas-1-i))

	for bloques_caracteristicas in lista_caracteristicas_v2:
		for i in range(dias_caracteristicas):
			for car in bloques_caracteristicas:
				columnas.append(car + '_' + str(dias_caracteristicas-1-i))
	dataset.columns = columnas

	dataset = pd.concat([dataset, df_CROCO_Dacumi['Dacuminata'+zona].iloc[dias_prediccion+(dias_caracteristicas-1):].reset_index().rename(columns={'Dacuminata'+zona:'Dacuminata'+zona+'_output'}).iloc[:,1]], axis=1)
	reg_nuls = dataset.isnull().sum()

	for label, content in reg_nuls.items():
		if content > df_CROCO_Dacumi.shape[0] - df_CROCO_Dacumi.shape[0]*0.1:
			dataset.pop(label)
	dataset = dataset.dropna(axis='rows')
	return dataset

def split_datasets(dataset, zona):
	list_temp = []; list_sal = []; list_ui = []; list_dacumi = []; list_u = []; list_v = []; list_bv = []; list_chl =[]
	for elem in dataset:
		if elem[0] == 'T': list_temp.append(elem)
		elif elem[0] == 'S': list_sal.append(elem)
		elif elem[0] == 'U' and elem[1] == 'I': list_ui.append(elem)
		elif elem[0] == 'D' and elem[-1] != 't': list_dacumi.append(elem)
		elif elem[0] == 'U': list_u.append(elem)
		elif elem[0] == 'V': list_v.append(elem)
		elif elem[0] == 'C': list_chl.append(elem)
		elif elem[0] == 'B' or elem[0] == 'd': list_bv.append(elem)

	return dataset[list_temp], dataset[list_sal], dataset[list_ui], dataset[list_dacumi], dataset[list_u], dataset[list_v], dataset[list_bv], dataset[list_chl], dataset['Month'], dataset['Dacuminata'+zona+'_output']

def _featuresSelection(dataset, zona, n):

	selector = feature_selection.SelectKBest(
		similarity=stats.PearsonCorr(),
		k=2
	)
	dataset_aux = dataset.copy()
	y = dataset_aux.pop('Dacuminata' + zona + '_output')
	for xi, yi, in stream.iter_pandas(dataset_aux, y):
		selector = selector.learn_one(xi, yi)

	feature_importances=[]
	for element in selector.leaderboard:
		feature_importances.append(abs(selector.leaderboard[element]))

	output = np.zeros(len(feature_importances))
	f_aux = feature_importances.copy()
	for i in range(len(output)):
		max_v = max(f_aux)
		max_idx = f_aux.index(max_v)
		f_aux[max_idx] = 0
		output[max_idx] = i + 1

	lista_caracteristicas = []
	labels = list(dataset_aux.keys())
	for i in range(n):
		idx = np.where(output == i+1)
		lista_caracteristicas.append(labels[idx[0][0]])

	return dataset_aux[lista_caracteristicas], y

def regressionModel(X,y,kwargs, apply_pca=True):
	if kwargs[1][-2] == 's': #Stream learning
		return _streamLearningModel(X,y, kwargs[0], kwargs[1], kwargs[2], apply_pca)
	else:					#Batch learning
		return _batchLearningModel(X,y, kwargs[1], kwargs[2], apply_pca)

def _streamLearningModel(x,y,dias_prediccion,modelo, kwargs, apply_pca=True):

	if modelo == 'knn_sl':
		model = (preprocessing.MinMaxScaler() | neighbors.KNNRegressor(n_neighbors=kwargs[0]))

	if modelo == 'hatr_sl':
		model = (preprocessing.MinMaxScaler() | tree.HoeffdingAdaptiveTreeRegressor(grace_period=kwargs[0], delta=kwargs[1], model_selector_decay=kwargs[2], max_depth=kwargs[3], tau=kwargs[4], seed = 42))
	if modelo == 'htr_sl':
		model = (preprocessing.MinMaxScaler() | tree.HoeffdingTreeRegressor(grace_period=kwargs[0], delta=kwargs[1], model_selector_decay=kwargs[2], max_depth=kwargs[3], tau=kwargs[4]))

	metric_r2 = metrics.R2()
	metric_mae = metrics.MAE()
	metric_mse = metrics.MSE()
	metric_rmse = metrics.RMSE()
	aux_list = []
	outputs = []
	contador = 0

	if apply_pca:
		pca = PCA(n_components=0.999, svd_solver='full')
		x_preTrain = pca.fit_transform(x.iloc[0:365, :])
		y_preTrain = y.iloc[0:365]
		x_train_test = pca.transform(x.iloc[365:, :])
		y_train_test = y.iloc[365:]
		n_components = pca.n_components_
	else:
		x_preTrain = x.iloc[0:365, :]
		y_preTrain = y.iloc[0:365]
		x_train_test = x.iloc[365:, :]
		y_train_test = y.iloc[365:]
		n_components = x.shape[1]

	for xi, yi in stream.iter_pandas(pd.DataFrame(x_preTrain), y_preTrain, shuffle=True, seed=42):
		y_pred = model.predict_one(xi)
		model.learn_one(xi, y_pred)

	for xi, yi in stream.iter_pandas(pd.DataFrame(x_train_test), y_train_test):
		y_pred = model.predict_one(xi)

		if contador >= len(y_train_test) - 365:
			metric_r2.update(yi, y_pred)
			metric_mae.update(yi, y_pred)
			metric_mse.update(yi, y_pred)
			metric_rmse.update(yi, y_pred)
			outputs.append(y_pred)

		aux_list.append((xi, yi))
		if contador+1 >= dias_prediccion:
			model.learn_one(aux_list[contador - (dias_prediccion - 1)][0], aux_list[contador - (dias_prediccion - 1)][1])
		contador += 1

	return [metric_r2.get(), metric_mae.get(), metric_mse.get(), metric_rmse.get()], outputs, n_components

def _batchLearningModel(x,y, model, args, apply_pca=True):

	if apply_pca:
		pca = PCA(n_components=0.999, svd_solver='full')
		pca.fit(x.iloc[0:365, :])
		x_train = pca.transform(x.iloc[:-365, :])
		y_train = y.iloc[:-365]
		x_test = pca.transform(x.iloc[-365:, :])
		y_test = y.iloc[-365:]
		n_components = pca.n_components_
	else:
		x_train = x.iloc[:-365, :]
		y_train = y.iloc[:-365]
		x_test = x.iloc[-365:, :]
		y_test = y.iloc[-365:]
		n_components = x.shape[1]

	scaler = MinMaxScaler()
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)

	output = []

	if model != 'mlp_bl':
		if model == 'rf_bl':
			regr = RandomForestRegressor(n_estimators=args[2], criterion=args[0], max_depth=args[1], random_state=42)

		if model == 'knn_bl':
			regr = KNeighborsRegressor(n_neighbors=args[0])

		if model == 'svr_bl':
			regr = SVR(kernel=args[0], C=args[1], epsilon=args[2], degree=args[3])

		regr.fit(x_train, y_train)
		#metric = regr.score(x_test, y_test)
		y_pred = regr.predict(x_test)
		r2 = r2_score(y_test, y_pred)
		mae = mean_absolute_error(y_test, y_pred)
		mse = mean_squared_error(y_test, y_pred)
		rmse = mean_squared_error(y_test, y_pred, squared=False)
		output = y_pred
		metric = [r2, mae, mse, rmse]

	if model == 'mlp_bl':

		neuronas_c1 = args[0][0]
		if len(args[0])==2:
			neuronas_c2 = args[0][1]

		inp = x_train.shape[1]

		list_metric_r2 = []
		list_metric_mae = []
		list_metric_mse = []
		list_metric_rmse = []
		for i in range(10):
			modelo = tf.keras.models.Sequential()

			modelo.add(tf.keras.layers.Dense(neuronas_c1, input_dim=inp, activation='relu'))
			if len(args[0])==2:
				modelo.add(tf.keras.layers.Dense(neuronas_c2, activation='relu'))
			modelo.add(tf.keras.layers.Dense(1))

			optimizer = tf.keras.optimizers.RMSprop(learning_rate=args[1])
			modelo.compile(loss='mse', optimizer=optimizer)

			callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
			modelo.fit(x=x_train,
					   y=y_train,
					   batch_size=64,
					   epochs=500,
					   validation_split=0.1,
					   validation_freq=1,
					   callbacks=[callback],
					   verbose=0)

			y_pred = modelo.predict(x_test)
			r2 = r2_score(y_test, y_pred)
			mae = mean_absolute_error(y_test, y_pred)
			mse = mean_squared_error(y_test, y_pred)
			rmse = mean_squared_error(y_test, y_pred, squared=False)
			list_metric_r2.append(r2)
			list_metric_mae.append(mae)
			list_metric_mse.append(mse)
			list_metric_rmse.append(rmse)

			for i in range(len(y_test)):
				output.append(modelo.predict(x_test[i, :].reshape(1, -1), verbose=0)[0])

			tf.keras.backend.clear_session()
			del modelo

		metric = [np.mean(list_metric_r2), np.mean(list_metric_mae), np.mean(list_metric_mse), np.mean(list_metric_rmse)]

	return metric, output, n_components

def getParams(model):

	if model == 'knn_sl' or model == 'knn_bl':
		k = random.sample([1, 3, 5, 7, 9, 11], k=1)[0]
		params = [k, None, None, None, None]

	if model == 'htr_sl' or model == 'hatr_sl':
		grace_period = random.randint(1, 365)
		delta = random.sample([1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10], k=1)[0]
		model_selector_decay = random.uniform(0, 1)
		max_depth = random.sample([2, 4, 10, 20, 50, None], k=1)[0]
		tau = random.sample([0.01, 0.05, 0.1], k=1)[0]
		params = [grace_period, delta, model_selector_decay, max_depth, tau]

	if model == 'svr_bl':
		kernel = random.sample(['linear', 'poly', 'rbf'], k=1)[0]
		degree = random.randint(1, 3)
		c = random.sample([0.001, 0.01, 0.05, 0.1, 1, 10], k=1)[0]
		e = random.uniform(0, 1)
		params = [kernel, c, e, degree, None]

	if model == 'mlp_bl':
		learning_rate = random.sample([1e-1, 1e-2, 1e-3, 1e-4, 1e-5], k=1)[0]
		n_neurons = \
			random.sample([[2], [4], [8], [10], [2, 2], [4, 2], [10, 2], [10, 10], [10, 2], [16, 8], [32, 16], [32, 8]],
						  k=1)[0]
		params = [n_neurons, learning_rate, None, None, None]
	if model == 'rf_bl':
		criterion = random.sample(['squared_error', 'friedman_mse', 'poisson'], k=1)[0]
		max_depth = random.sample([2, 4, 10, 20, 50, None], k=1)[0]
		n_estimators = random.randint(3, 1000)
		params = [criterion, max_depth, n_estimators, None, None]

	return params
