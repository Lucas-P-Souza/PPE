
# Importation des bibliothèques nécessaires
import numpy as np  # Pour les calculs numériques et la manipulation de tableaux
import pandas as pd  # Pour lire et manipuler les données CSV
from scipy.io.wavfile import write  # Pour écrire des fichiers audio WAV
import os  # Pour la gestion des chemins de fichiers
from scipy.interpolate import interp1d  # Pour l'interpolation des données

def generate_audio_from_csv(csv_path=None, output_path=None):
	# Si aucun chemin CSV n'est fourni, utilise le chemin par défaut
	if csv_path is None:
		# Construit le chemin vers le fichier CSV dans le dossier back_end/results/plots
		csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots', 'string_positions.csv')
		csv_path = os.path.normpath(csv_path)
	
	# Lecture du fichier CSV contenant les positions de la corde
	if not os.path.isfile(csv_path):
		raise FileNotFoundError(
			f"Fichier CSV introuvable: {csv_path}.\n"
			"Assure-toi que la simulation a bien généré 'string_positions.csv' dans 'digital_twin/results/plots'."
		)
	data = pd.read_csv(csv_path)
	
	# Extraction de la colonne temps
	time = data['t'].values
	
	# Calcul de l'indice du point milieu de la corde
	mid_index = len([col for col in data.columns if col.startswith('u_')]) // 2
	
	# Construction du nom de la colonne du point milieu (ex: 'u_25')
	mid_col = f'u_{mid_index}'
	
	# Extraction des positions du point milieu au cours du temps
	position = data[mid_col].values
	
	# Normalisation des positions entre -1 et 1 pour le format audio
	position_norm = position / np.max(np.abs(position))
	
	# Définition de la fréquence d'échantillonnage audio standard (44.1 kHz)
	sample_rate = 44100
	
	# Calcul de la durée totale de la simulation
	duration = time[-1] - time[0]
	
	# Calcul du nombre d'échantillons nécessaires pour l'audio
	n_samples = int(duration * sample_rate)
	
	# Création d'un nouveau vecteur temps avec la fréquence d'échantillonnage audio
	new_time = np.linspace(time[0], time[-1], n_samples)
	
	# Création d'un interpolateur linéaire pour rééchantillonner les données
	interpolator = interp1d(time, position_norm, kind='linear')
	
	# Application de l'interpolation pour obtenir les données audio
	wave_data = interpolator(new_time)
	
	# Conversion en entiers 16 bits pour le format WAV (plage: -32767 à 32767)
	wave_int16 = np.int16(wave_data * 32767)
	
	# Définition du dossier de sortie (remonte d'1 niveau puis va dans results/plots)
	output_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results', 'plots'))
	
	# Création du dossier de sortie s'il n'existe pas
	os.makedirs(output_dir, exist_ok=True)
	
	# Si aucun chemin de sortie n'est fourni, utilise le chemin par défaut
	if output_path is None:
		output_path = os.path.join(output_dir, 'string_middle.wav')
	
	# Écriture du fichier audio WAV
	write(output_path, sample_rate, wave_int16)
	
	# Affichage d'un message de confirmation avec le chemin du fichier créé
	print(f'Son généré au milieu de la corde : {output_path}')
	
	# Retour du chemin du fichier créé
	return output_path

def generate_multiple_positions_audio(csv_path=None):
	"""
	Génère des fichiers audio pour différentes positions sur la corde
	pour comparer les timbres selon le point d'écoute
	"""
	# Si aucun chemin CSV n'est fourni, utilise le chemin par défaut
	if csv_path is None:
		csv_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'plots', 'string_positions.csv')
		csv_path = os.path.normpath(csv_path)
	
	# Lecture du fichier CSV
	if not os.path.isfile(csv_path):
		raise FileNotFoundError(
			f"Fichier CSV introuvable: {csv_path}.\n"
			"Assure-toi que la simulation a bien généré 'string_positions.csv' dans 'digital_twin/results/plots'."
		)
	data = pd.read_csv(csv_path)
	time = data['t'].values
	
	# Trouver toutes les colonnes de positions (u_0, u_1, u_2, ...)
	position_columns = [col for col in data.columns if col.startswith('u_')]
	total_points = len(position_columns)
	
	# Définir les positions d'intérêt (début, 1/4, milieu, 3/4, fin)
	positions_to_test = {
		'debut': 5,  # Près du début (évite les bords fixes)
		'quart': total_points // 4,  # Au quart de la corde
		'milieu': total_points // 2,  # Au milieu
		'trois_quarts': 3 * total_points // 4,  # Aux trois quarts
		'fin': total_points - 6  # Près de la fin (évite les bords fixes)
	}
	
	# Paramètres audio
	sample_rate = 44100
	duration = time[-1] - time[0]
	n_samples = int(duration * sample_rate)
	new_time = np.linspace(time[0], time[-1], n_samples)
	
	# Dossier de sortie - utilise le dossier back_end/results/plots
	output_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results', 'plots'))
	os.makedirs(output_dir, exist_ok=True)
	
	# Générer un fichier audio pour chaque position
	for position_name, point_index in positions_to_test.items():
		# Nom de la colonne correspondante
		col_name = f'u_{point_index}'
		
		# Vérifier que la colonne existe
		if col_name in data.columns:
			# Extraction des positions pour ce point
			position = data[col_name].values
			
			# Normalisation
			position_norm = position / np.max(np.abs(position))
			
			# Interpolation
			interpolator = interp1d(time, position_norm, kind='linear')
			wave_data = interpolator(new_time)
			
			# Conversion en format audio
			wave_int16 = np.int16(wave_data * 32767)
			
			# Nom du fichier de sortie
			output_path = os.path.join(output_dir, f'string_{position_name}.wav')
			
			# Sauvegarde
			write(output_path, sample_rate, wave_int16)
			
			print(f'Son généré pour la position {position_name} (point {point_index}): {output_path}')
	
	print(f'\n🎵 {len(positions_to_test)} fichiers audio générés!')
	print('Tu peux maintenant écouter et comparer les différents timbres selon la position sur la corde.')

# Exécution automatique lors du lancement du script
if __name__ == "__main__":
	print("🎼 Génération de fichiers audio pour différentes positions sur la corde...")
	generate_multiple_positions_audio()

