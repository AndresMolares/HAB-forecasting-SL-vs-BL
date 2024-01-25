import warnings
import functions as func
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from csv import writer
import argparse

parser = argparse.ArgumentParser(description='Process some params.')
parser.add_argument('-c', '--count')
args = parser.parse_args()

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
#sys.setrecursionlimit(9999)
#random.seed(1)


################################################################
################ Code ##########################################
################################################################

def save_results(nombre_archivo_salida, model, score, params, zona, dias_prediccion, n_features):

	with open(nombre_archivo_salida, 'a', newline='') as f_object:
		writer_object = writer(f_object)
		writer_object.writerow(
			[model, score[0], score[1], score[2], score[3], params[0], params[1],
			 params[2], params[3], params[4], zona, dias_prediccion, n_features])
		f_object.close()

def greed_search(dataset, zona, nombre_archivo_salida, dias_prediccion):

	models = ['knn_sl', 'htr_sl', 'hatr_sl', 'svr_bl', 'knn_bl', 'mlp_bl', 'rf_bl']
	for model in models:
		if model != models[-1]:
			print(model, end='')
		else:
			print(model)

		dataset_aux = dataset.copy()
		y = dataset_aux.pop('Dacuminata' + zona + '_output')
		X = dataset_aux

		if model == 'knn_sl' or model == 'knn_bl':
			for k in [1, 3, 5, 7, 9, 11]:
				params = [k, None, None, None, None]
				score, _, n_features = func.regressionModel(X, y, [dias_prediccion, model, params], False)
				save_results(nombre_archivo_salida, model, score, params, zona, dias_prediccion, n_features)

		if model == 'htr_sl' or model == 'hatr_sl':
			for grace_period in [7, 14, 30, 180, 365]:
				for delta in [1e-5, 1e-6, 1e-7, 1e-8]:
					for model_selector_decay in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
						for tau in [0.01, 0.05, 0.1]:
							params = [grace_period, delta, model_selector_decay, None, tau]
							score, _, n_features = func.regressionModel(X, y, [dias_prediccion, model, params], False)
							save_results(nombre_archivo_salida, model, score, params, zona, dias_prediccion, n_features)

		if model == 'svr_bl':
			for kernel in ['linear', 'poly', 'rbf']:
				for c in [0.001, 0.01, 0.05, 0.1, 1, 10]:
					for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
						if kernel == 'poly':
							for degree in [1, 2, 3]:
								params = [kernel, c, e, degree, None]
								score, _, n_features = func.regressionModel(X, y, [dias_prediccion, model, params], False)
								save_results(nombre_archivo_salida, model, score, params, zona, dias_prediccion, n_features)
						else:
							params = [kernel, c, e, 1, None]
							score, _, n_features = func.regressionModel(X, y, [dias_prediccion, model, params], False)
							save_results(nombre_archivo_salida, model, score, params, zona, dias_prediccion, n_features)

		if model == 'mlp_bl':
			for learning_rate in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
				for n_neurons in [2], [4], [8], [10], [2, 2], [4, 2], [10, 2], [10, 10], [10, 2], [16, 8], [32, 16], [32, 8]:
					params = [n_neurons, learning_rate, None, None, None]
					score, _, n_features = func.regressionModel(X, y, [dias_prediccion, model, params], False)
					save_results(nombre_archivo_salida, model, score, params, zona, dias_prediccion, n_features)

		if model == 'rf_bl':
			for criterion in ['squared_error', 'friedman_mse', 'poisson']:
				for max_depth in [2, 4, 10, 20, 50, None]:
					params = [criterion, max_depth, 1000, None, None]
					score, _, n_features = func.regressionModel(X, y, [dias_prediccion, model, params], False)
					save_results(nombre_archivo_salida, model, score, params, zona, dias_prediccion, n_features)


	r2_base = r2_score(dataset['Dacuminata' + zona + '_output'], dataset['Dacuminata' + zona + '_0'])
	mae = mean_absolute_error(dataset['Dacuminata' + zona + '_output'], dataset['Dacuminata' + zona + '_0'])
	mse = mean_squared_error(dataset['Dacuminata' + zona + '_output'], dataset['Dacuminata' + zona + '_0'])
	rmse = mean_squared_error(dataset['Dacuminata' + zona + '_output'], dataset['Dacuminata' + zona + '_0'], squared=False)
	with open(nombre_archivo_salida, 'a', newline='') as f_object:
		writer_object = writer(f_object)
		writer_object.writerow(['base_line', r2_base, mae, mse, rmse, None, None, None, None, None, zona, dias_prediccion, None])
		f_object.close()

################################################################
############ MAIN ##############################################
################################################################

#########Configuracion####################
zonas = {
		 'P2':['P8','P9','SP1','SP2'],
		 'P4':['P2','P5','SP2','SP3','SP4'],
		 'V1':['V2','V5','V6','SV2','SV3','SV4'],
		 'V4':['V2','V3','SV1','SV2'],
		 'A3':['A1','A5','SA1','SA2'],
		 'A8':['A0','A4','A7','SA3','SA4','SA5']
		}

args_count = int(args.count)

if args_count < 8:
	prediction_days = args_count
	zona = 'P2'
elif args_count < 15:
	prediction_days = args_count - 7
	zona = 'P4'
elif args_count < 22:
	prediction_days = args_count - 14
	zona = 'V1'
elif args_count < 29:
	prediction_days = args_count - 21
	zona = 'V4'
elif args_count < 36:
	prediction_days = args_count - 28
	zona = 'A3'
else:
	prediction_days = args_count - 35
	zona = 'A8'

information_days = 7
nombre_archivo_salida = './results/model_comparation_greed_search_' + zona + '_' + str(prediction_days) + '.csv'

with open(nombre_archivo_salida, 'a', newline='') as f_object:
	writer_object = writer(f_object)
	writer_object.writerow(['MODEL', 'R2', 'MAE', 'MSE', 'RMSE', 'P1', 'P2', 'P3', 'P4', 'P5', 'STATION', 'PD', 'NC'])
	f_object.close()
#########Code######################################################################################################

aux = lambda x: math.sin(math.radians((180 / 11) * x - (180 / 11)))

#main bucle #######################################################################################################

zonas_aux = zonas[zona]
dataset, l_carac_v1, l_carac_v2, df_CROCO_Dacumi = func.obtenerDataset('./data/DailyUI_ncep.csv', './data/STA_CROCO_DacumiReal2013a2019_gapfree.csv', './data/SECT_CROCO2.csv', prediction_days, information_days, zona, zonas_aux)
dataset = func.reetiquetarDataset(dataset, l_carac_v1, l_carac_v2, df_CROCO_Dacumi, zona,prediction_days, information_days)
dataset['Month'] = dataset['Month'].apply(aux)
greed_search(dataset, zona, nombre_archivo_salida, prediction_days)

###################################################################################################################
