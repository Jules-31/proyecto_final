import os
import torch #Procesamiento de audio y redes neuronales
import torchaudio #Procesamiento de audio y redes neuronales
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from glob import glob #Encontrar los nombres de los archivos
from tqdm import tqdm #Barras de prograso
import warnings

warnings.filterwarnings('ignore')

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Usar GPU o CPU, acelera el entrenamiento si hay muchos datos
SAMPLE_RATE = 22050 #Frecuencia de muestreo en kHz
MAX_LEN = 3  # segundos
BATCH_SIZE = 32 #Npumero de muestras de entrenamiento procesadas antes de actualizar los pesos (balance memoria y rgadiente)
EPOCHS = 5 #Épocas de entrenamiento
LEARNING_RATE = 3e-4 #Controlar el tamaño de los pasos al actualizar los pesos
DATA_DIR = r"C:\Users\saray\Downloads\oyopf\all-sample-des"  # Cambiar según ruta
N_FFT = 2048  # Tamaño de la FFT

"""
###############################################################################
# Atribuciones y Licencias de Código Reutilizado
###############################################################################

Este proyecto utiliza implementaciones adaptadas de los siguientes trabajos:

1. Procesamiento de Audio (AudioProcessor):
   - Basado en: Sreedhar, D. (2021). Music-Instrument-Recognition. GitHub.
     Licencia: MIT (https://github.com/dhivyasreedhar/Music-Instrument-Recognition)
   - Inspirado por: Siddhant, O. (2022). Musical-Instruments-Classification-CNN. Kaggle.
     Licencia: Apache 2.0 (https://www.kaggle.com/code/siddhantojha17/...)

2. Dataset y Aumento de Datos (InstrumentDataset):
   - Adaptado de: GuitarsAI. (2020). BasicsMusicalInstrumClassifi. GitHub.
     Licencia: Apache 2.0 (https://github.com/GuitarsAI/BasicsMusicalInstrumClassifi)

3. Arquitectura CNN (InstrumentCNN):
   - Implementación derivada de: 
     - Musikalkemist. (2021). DeepLearningForAudioWithPython. GitHub.
       Licencia: MIT (https://github.com/musikalkemist/DeepLearningForAudioWithPython)
     - Sreedhar, D. (2021). Music-Instrument-Recognition. GitHub.
       Licencia: MIT (https://github.com/dhivyasreedhar/Music-Instrument-Recognition)

4. Visualización (TrainingVisualizer):
   - Combinación de técnicas de:
     - Anirudh, S. (2021). Music-Instrument-Classification. GitHub.
       Licencia: MIT (https://github.com/anirudhs123/Music-Instrument-Classification)
     - GuitarsAI (2020). BasicsMusicalInstrumClassifi. GitHub.
       Licencia: Apache 2.0

5. Entrenamiento (train_model):
   - Métodos adaptados de:
     - Chulev, J. (2022). AI-Instrument-Classification. GitHub.
       Licencia: GPL-3.0 (https://github.com/JoanikijChulev/AI-Instrument-Classification)

###############################################################################
# Notas de Licencia
# - Este proyecto se distribuye bajo la licencia MIT.
# - Verifique las licencias originales antes de redistribuir código derivado.
###############################################################################
"""

# 1. Procesamiento de Audio con FFT
class AudioProcessor:
    """
    Clase para procesamiento de audio usando FFT directa.
    Implementación adaptada de:
    - Sreedhar, D. (2021). Music-Instrument-Recognition. GitHub. 
        https://github.com/dhivyasreedhar/Music-Instrument-Recognition
    - Siddhant, O. (2022). Musical-Instruments-Classification-CNN. Kaggle.
        https://www.kaggle.com/code/siddhantojha17/musical-instruments-classification-cnn

    """
    def __init__(self, sample_rate=22050, n_fft=2048):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        # Ventana Hann para la FFT
        self.window = torch.hann_window(n_fft)
        
    def magnitude_fft(self, waveform):
        """Calcula la magnitud del espectro FFT."""
        windowed = waveform * self.window.to(waveform.device) #Aplica la ventana Hann
        fft = torch.fft.rfft(windowed, n=self.n_fft) #tomar solo fft positiva
        mag = torch.abs(fft) #magnitud fft
        mag = torch.log1p(mag) #usar log(1+x) compresión
        return mag

    def process(self, waveform, sample_rate):
        """
        Procesa una forma de onda: re-muestrea, normaliza y calcula el espectro FFT.
        """
        #Misma frecuencia
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)  
            waveform = resampler(waveform)
        #Monoaudio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        #Duración    
        target_samples = int(self.sample_rate * MAX_LEN)
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]
        else:
            pad_amount = target_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_amount))
        
        waveform = waveform / waveform.abs().max()
        hop_length = self.n_fft // 2
        n_segments = (waveform.shape[1] - self.n_fft) // hop_length + 1
        spectrogram = []
        
        for i in range(n_segments): 
            start = i * hop_length 
            end = start + self.n_fft 
            segment = waveform[:, start:end]
            fft_mag = self.magnitude_fft(segment)
            spectrogram.append(fft_mag)
        spectrogram = torch.stack(spectrogram, dim=2)  
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8) 
        return spectrogram 

# 2. Dataset
class InstrumentDataset(Dataset):
    """
    Basado en:
    - GuitarsAI. (2020). BasicsMusicalInstrumClassifi. GitHub.
      https://github.com/GuitarsAI/BasicsMusicalInstrumClassifi
    
    Args:
        data_dir (str): Ruta al directorio con subcarpetas por clase.
        augment (bool): Si se aplica aumento de datos (desplazamiento temporal).
        max_samples_per_class (int): Límite de muestras por clase (evita desbalanceo).
    """
    #Cambio para usar tanto .mp3 como .wav
    def __init__(self, data_dir, augment=False, max_samples_per_class=200): #Augment: aumento dedatos, max límite de audios por clase
        self.processor = AudioProcessor(SAMPLE_RATE) 
        self.augment = augment
        self.samples = [] 
        self.label_map = {} 
        self.inverse_map = {} 
        
        #Analizar subcarpetas
        class_dirs = sorted(glob(os.path.join(data_dir, "*"))) 
        for class_idx, class_dir in enumerate(class_dirs): 
            class_name = os.path.basename(class_dir)
            self.label_map[class_idx] = class_name 
            self.inverse_map[class_name] = class_idx 
            
            #Para que lea mp3 o wav
            audio_files = glob(os.path.join(class_dir, "*.mp3")) + glob(os.path.join(class_dir, "*.wav"))
            audio_files = audio_files[:max_samples_per_class] #Balanceo de clases
            
            for audio_file in tqdm(audio_files, desc=f"Cargando {class_name}"): #Barra de progreso de carga de audios
                self.samples.append((audio_file, class_idx)) 
        
        print(f"\nDataset cargado: {len(self.samples)} muestras, {len(self.label_map)} clases")

    def __len__(self):
        return len(self.samples) #Devolver las muestras en el dataset 

    def __getitem__(self, idx): #Obteenr y procesar las muestras
        """
        Devuelve una muestra procesada lista para ser alimentada a un modelo.

        Args:
            idx (int): Índice de la muestra.

        Returns:
            Tuple[Tensor, int]: (espectrograma, etiqueta)
        """
        audio_path, label = self.samples[idx] #Obtener ruta de audio y etiqueta
        try:
            waveform, sample_rate = torchaudio.load(audio_path) #carga el archivo de audio, devuelve un tensor con los adatos de audio y la frecuencia de muestreo
            spec = self.processor.process(waveform, sample_rate) #Procesa el audio crudo a espectrogramas convencionales
            
            #Aumentar datos para entrenar
            if self.augment and np.random.random() < 0.5: #Aumenta los datos si hay 50% de probabilidad de usar un desplazamiento temporal en el espectrograma, para reconocer patrones independiente de la posición temporal
                spec = torch.roll(spec, shifts=np.random.randint(-10, 10), dims=2) #Desplzar el espectrograma por el eje temporal
            #Asegurar las dimensiones
            if spec.dim() == 2: #ASegurar que el espectrograma tenga tres dimensiones
                spec = spec.unsqueeze(0) #Añade una dimensión si tiene solo 2
            return spec, label
            
        except Exception as e:
            print(f"Error procesando {audio_path}: {str(e)}")
            #Si hay errores, el espectograma es blanco
            dummy = torch.zeros((1, self.processor.n_mels, int(SAMPLE_RATE * MAX_LEN / 512) + 1))
            return dummy, 0

    def get_class_weights(self):
        """
        Calcula pesos por clase inversamente proporcionales a su frecuencia.
        Esto permite balancear clases desequilibradas durante el entrenamiento.

        Returns:
            Tensor: Pesos normalizados por clase (float32).
        """
        #Fragmento basado en:
        #https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/8-%20Training%20a%20neural%20network%3A%20Implementing%20back%20propagation%20from%20scratch/code/mlp.py
        #Se tomó como referencia para saber cómo hacer el ajuste de los pesos por cada clase
        counts = np.bincount([label for _, label in self.samples])  #Cuenta número de uestras por clase, extrae las etiquetas y cuneta cuántas veces aparece
        counts = np.where(counts == 0, 1, counts) #Evitar divisiones entre 0, reemplazar 0 por 1
        weights = 1. / counts # calcula pesos inversamente proporcionales a la frecuencia de cada clase, a menor muestra mayor peso
        weights = weights / weights.sum() * len(weights) #Normalizar
        return torch.tensor(weights, dtype=torch.float32) #Convierte los pesos a un tensor para usar como funicón de perdida

# 3. Modelo CNN
class InstrumentCNN(nn.Module):
    """CNN para clasificación de espectrogramas de convencionales.
    
    Arquitectura:
        - 3 bloques Conv2D + BatchNorm + ReLU + MaxPool.
        - Capa lineal final con Dropout.
    
    Basado en:
    - Sreedhar, D. (2021). Music-Instrument-Recognition. GitHub.
      https://github.com/dhivyasreedhar/Music-Instrument-Recognition/blob/main/cnn.ipynb
    
    Args:
        num_classes (int): Número de clases de instrumentos.
    """
    #Cambios usando librería de pytorch para convolución, normalización, función de activación, agrupaciones, probabilidades
    def __init__(self, num_classes):
        super().__init__() #Llama nn.Module
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """Propagación hacia adelante del modelo CNN.
        
        Procesa el tensor de entrada a través de las capas convolucionales y lineales
        para generar logits de clasificación. Adaptado de la implementación de:
        - Musikalkemist. (2021). DeepLearningForAudioWithPython. GitHub.
        https://github.com/musikalkemist/DeepLearningForAudioWithPython

        Args:
            x (torch.Tensor): Tensor de entrada con forma [batch_size, 1, n_mels, tiempo].
                            Si tiene 3 dimensiones, se añade automáticamente la dimensión del canal.

        Returns:
            torch.Tensor: Logits de clasificación con forma [batch_size, num_classes].
        """
        #Se utilizó como guía para definir la propagación, se ajustó de acuerdo al formato del código
        if x.dim() == 3: #Si la entrada es de 3 dimensiones 
            x = x.unsqueeze(1) 
        x = self.features(x) #Capas convolucionales, extraer características 
        x = x.view(x.size(0), -1) #Compacta el resultado que entra a classifier 
        return self.classifier(x) #Pasa por cada capa para obtener las predicciones

# 4. Visualización
class TrainingVisualizer:
    """Visualiza métricas de entrenamiento y resultados del modelo.
    
    Implementación basada en:
    - GuitarsAI. (2020). BasicsMusicalInstrumClassifi. GitHub.
      https://github.com/GuitarsAI/BasicsMusicalInstrumClassifi
    - Sreedhar, D. (2021). Music-Instrument-Recognition. GitHub.
      https://github.com/dhivyasreedhar/Music-Instrument-Recognition

    Args:
        label_map (dict): Mapeo de índices a nombres de clases (ej: {0: 'guitar', 1: 'piano'}).
    """
    #Se toma la forma más eficaz y factible de visualizar y comprender los datos, con dos gráficas que muestras
    #la evolución de aciertos y pérdidas, y usando matrices de confución para conocer cómo es la clasificación
    def __init__(self, label_map):
        self.label_map = label_map 
        self.train_loss = [] 
        self.val_loss = []
        self.train_acc = [] 
        self.val_acc = []
        
    def update(self, epoch, tr_loss, val_loss, tr_acc, val_acc, model, val_loader):
        """
        Actualiza las métricas y genera visualizaciones cada 5 épocas o al final.

        Args:
            epoch (int): Número de época actual.
            tr_loss (float): Pérdida del conjunto de entrenamiento.
            val_loss (float): Pérdida del conjunto de validación.
            tr_acc (float): Precisión en entrenamiento.
            val_acc (float): Precisión en validación.
            model (nn.Module): Modelo entrenado.
            val_loader (DataLoader): Dataloader de validación.
        """
        self.train_loss.append(tr_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(tr_acc)
        self.val_acc.append(val_acc)
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1: 
            self._plot_metrics() 
            self._plot_confusion_matrix(model, val_loader) 

    def _plot_metrics(self):
        """Genera y guarda un gráfico de la evolución de pérdida y precisión."""
        plt.figure(figsize=(12, 5))
        
        #Gráfica de pérdida
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label='Entrenamiento', color="#551C8A",  linestyle=":")
        plt.plot(self.val_loss, label='Validación', color="#8A1C60DA",  linestyle="-")
        plt.title('Evolución de la Pérdida')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        #Gráfica de precisión
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc, label='Entrenamiento', color="#391C8A",  linestyle=":")
        plt.plot(self.val_acc, label='Validación', color="#8A1C3DDA",  linestyle="-")
        plt.title('Evolución de la Precisión')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metricsespectro.png')
        plt.close()
    
    def _plot_confusion_matrix(self, model, loader):
        """
        Genera y guarda la matriz de confusión junto con un reporte de clasificación.

        Args:
            model (nn.Module): Modelo entrenado.
            loader (DataLoader): Loader del conjunto de validación/test.
        """
        model.eval() 
        all_preds = [] 
        all_labels = [] 
        
        #Predicciones
        with torch.no_grad(): 
            for inputs, labels in loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs) 
                _, preds = torch.max(outputs, 1) 
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        #Matriz de confusión
        cm = confusion_matrix(all_labels, all_preds) 
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_map.values(), yticklabels=self.label_map.values())
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Verdaderos')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrixespectro.png')
        plt.close()
        
        print("\nReporte de Clasificación:")
        print(classification_report(all_labels, all_preds, target_names=self.label_map.values()))
        #Precision, exactitud al predecir Precision= true pos / (true pos + false pos)
        #Recall, capacidad para detectar los true positives de una clase Recall = true pos / (true pos + false neg)
        #f1-score, balance entre las anteriores, f1 = 2 * (precision*recall) / (precision+recall)
        #Support, número de muestras reales de cada clase por conjunto de prueba, para identificar resultados si quedan sesgados por clase 

# 5. Funciones de Entrenamiento
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, label_map):
    """Entrena un modelo CNN y evalúa su rendimiento.
    
    Basado en:
    - GuitarsAI. (2020). BasicsMusicalInstrumClassifi. GitHub.
      https://github.com/GuitarsAI/BasicsMusicalInstrumClassifi
    - Chulev, J. (2022). AI-Instrument-Classification. GitHub.
      https://github.com/JoanikijChulev/AI-Instrument-Classification

    Args:
        model (nn.Module): Modelo a entrenar.
        train_loader (DataLoader): Loader de datos de entrenamiento.
        val_loader (DataLoader): Loader de datos de validación.
        criterion (nn.Module): Función de pérdida (ej: CrossEntropyLoss).
        optimizer (torch.optim): Optimizador (ej: AdamW).
        scheduler (torch.optim.lr_scheduler): Planificador de tasa de aprendizaje.
        num_epochs (int): Número de épocas de entrenamiento.
        label_map (dict): Mapeo de índices a nombres de clases.

    Returns:
        nn.Module: Modelo entrenado con los mejores pesos guardados en 'best_model.pth'.
    """
    #Se utilizaron como referencia de qué parámetros se debían usar para el entrenamiento y cómo
    visualizer = TrainingVisualizer(label_map) 
    best_acc = 0.0 
    epoch_times = [] 
    
    
    for epoch in range(num_epochs):
        start_time = time.time() 
        model.train() 
        running_loss = 0.0 
        running_corrects = 0 
        
        #Entrenamiento
        for inputs, labels in train_loader: 
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad() 
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            
            _, preds = torch.max(outputs, 1) 
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        #Cálculo métricas
        train_loss = running_loss / len(train_loader.dataset) 
        train_acc = running_corrects.double() / len(train_loader.dataset) 
        
        #Validaciones
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad(): 
            for inputs, labels in val_loader: 
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs) #Predicciones del modelo
                loss = criterion(outputs, labels) 
                _, preds = torch.max(outputs, 1) 
                val_loss += loss.item() * inputs.size(0) 
                val_corrects += torch.sum(preds == labels.data) 
        
        val_loss = val_loss / len(val_loader.dataset) #Calcula pérdidas
        val_acc = val_corrects.double() / len(val_loader.dataset) #Calcula aciertos
        
        #Actualizar schedueler y guardar mejor modelo
        if scheduler:
            scheduler.step(val_loss) 
        
        if val_acc > best_acc: 
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        #Visualización
        epoch_time = time.time() - start_time 
        epoch_times.append(epoch_time)
        avg_time = sum(epoch_times) / len(epoch_times) 
        remaining_time = avg_time * (num_epochs - epoch - 1) 
        
        visualizer.update(epoch, train_loss, val_loss, train_acc.item(), val_acc.item(), model, val_loader) #Actualizar gráficas y métricas
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
              f'Tiempo: {epoch_time:.1f}s | '
              f'ETA: {remaining_time/60:.1f}min')
    total_time = sum(epoch_times)
    avg_epoch_time = total_time / num_epochs
    print(f'\nEntrenamiento completado en {total_time/60:.1f} minutos')
    print(f'Tiempo promedio por época: {avg_epoch_time:.1f} segundos')
    
    return model

def create_dataloaders(data_dir, batch_size=32, val_split=0.2):
    """Crea DataLoaders balanceados para entrenamiento y validación.
    
    Args:
        data_dir (str): Ruta al directorio con subcarpetas por clase.
        batch_size (int): Tamaño del lote (default: 32).
        val_split (float): Proporción de datos para validación (default: 0.2).

    Returns:
        tuple: (train_loader, val_loader, label_map)
    """
    full_dataset = InstrumentDataset(data_dir, augment=True) 
    
    indices = list(range(len(full_dataset))) 
    labels = [full_dataset.samples[i][1] for i in indices] 
    
    train_indices, val_indices = train_test_split(indices, test_size=val_split, stratify=labels) 
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices) 
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices) 
    
    train_labels = [full_dataset.samples[i][1] for i in train_indices] 
    class_weights = full_dataset.get_class_weights() 
    sample_weights = class_weights[train_labels] 
    
    if (sample_weights <= 0).any(): #Asegurar ningún peso 0
        sample_weights = torch.clamp(sample_weights, min=1e-8) #Forzar valor mínimo
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True) 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) 
    
    return train_loader, val_loader, full_dataset.label_map

# 6. Clasificación de Audios Nuevos
def predict_audio(file_path, model_path='best_model.pth', threshold=0.6, show_spectrogram=True): #Threshold, minimo de confianza
    """
    Clasifica un archivo de audio y visualiza el espectrograma y la probabilidad por clase.

    Args:
        file_path (str): Ruta al archivo de audio (.wav).
        model_path (str): Ruta al modelo entrenado.
        threshold (float): Umbral de confianza para la predicción.
        show_spectrogram (bool): Si se desea mostrar el espectrograma.

    Returns:
        Tuple[str, float]: Clase predicha y confianza asociada.
    """
    if not hasattr(predict_audio, 'label_map'):  #Ejecutar una vez por sesión
        _, _, predict_audio.label_map = create_dataloaders(DATA_DIR, BATCH_SIZE) #Nombres de clases ordenados
        predict_audio.model = InstrumentCNN(len(predict_audio.label_map)).to(DEVICE) #Crear cnn con cantidad correcta de clases
        predict_audio.model.load_state_dict(torch.load(model_path)) #Cargar pesos entrenados del .pth
        predict_audio.model.eval() #Evalua modelo
        predict_audio.processor = AudioProcessor(SAMPLE_RATE) #Genera espectrograma

    
    try:
        waveform, sample_rate = torchaudio.load(file_path) #Cargar archivo de audio
        spec = predict_audio.processor.process(waveform, sample_rate) #Convertir onda en espectrograma de decibelios
        if spec.dim() == 2:
            spec = spec.unsqueeze(0).unsqueeze(0)  # [n_mels, time] → [1, 1, n_mels, time]
        elif spec.dim() == 3:
            spec = spec.unsqueeze(1)  # [1, n_mels, time] → [1, 1, n_mels, time]
        elif spec.dim() == 4:
            pass  # Ya está bien
        else:
            raise ValueError(f"Forma inesperada del espectrograma: {spec.shape}")

        spec = spec.to(DEVICE)
        
        with torch.no_grad():
                outputs = predict_audio.model(spec) #Obtiene logits, vectores
                probs = F.softmax(outputs, dim=1) #Convierte logit en probabilidad
                conf, pred = torch.max(probs, 1) #Indice clase mas probable y confianza
                conf = conf.item()
                pred_class = predict_audio.label_map[pred.item()] #Convertir a valores de py normales, float str
            
        print(f"\n Audio analizado: {os.path.basename(file_path)}")
        
        if conf >= threshold:
            print(f" Predicción: {pred_class} (Confianza: {conf:.2%})") #Si confianza supea umbral, muestra la clase, si no, la prediccion no es confiables
        else:
            print(f" Predicción incierta: {pred_class} (Confianza: {conf:.2%} < {threshold:.0%})") #Clases con su  probabilidad
        
        print("\nDistribución de probabilidades:")
        for i, prob in enumerate(probs.squeeze().cpu().numpy()):
            print(f"- {predict_audio.label_map[i]}: {prob:.2%}")
        
        if show_spectrogram:
            plt.figure(figsize=(10, 4))
            plt.imshow(spec.squeeze().cpu().numpy(), aspect='auto', origin='lower')
            plt.title(f"Espectrograma | Pred: {pred_class} ({conf:.2%})")
            plt.colorbar()
            plt.tight_layout()
            plt.show()
        
        return pred_class, conf
    
    except Exception as e:
        print(f"Error procesando el audio: {str(e)}")
        return None, None
    

# 7. Función Principal y Menú
def main():
    """
    Función principal que ejecuta el pipeline completo de entrenamiento.
    """
    print("=== CLASIFICADOR DE INSTRUMENTOS MUSICALES ===")
    print(f"Dispositivo: {DEVICE}")
    
    print("\nCargando dataset")
    train_loader, val_loader, label_map = create_dataloaders(DATA_DIR, BATCH_SIZE)
    
    model = InstrumentCNN(len(label_map)).to(DEVICE)
    print(f"\nModelo creado con {len(label_map)} clases")
    print(f"Total parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print("\nIniciando entrenamiento")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, label_map)
    
    print("\nEntrenamiento completado")
    print(f"Mejores pesos guardados en: best_model.pth")

if __name__ == "__main__":
    while True:
        print("MENÚ")
        print("1. Entrenar modelo")
        print("2. Clasificar un audio")
        print("3. Salir")
    
        choice = input("Seleccione una opción: ")
        if choice == "1":
            main()
        elif choice == "2":
            path = input("Ruta del archivo de audio: ")
            if os.path.exists(path):
                predict_audio(path)
            else:
                print(" El archivo no existe. Verifique la ruta.")
        elif choice == "3":
            print("Adiós")
        else:
            print("Opción no válida.")

