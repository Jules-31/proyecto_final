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

# 1. Procesamiento de Audio
#Audio  → Remuestreo → Mono → Padding → Normalización → MelSpectrogram → dB → Estandarización
class AudioProcessor:
    """
    Clase para procesamiento de audio usando STFT (espectro convencional)
    en lugar de MelSpectrogram.
    """
    def __init__(self, sample_rate=22050, n_fft=2048): #Frecuencia de muestreo & número de bandas de frecuencia
        self.sample_rate = sample_rate #Guardar la frecuencia de muestreo como atributo de clase
        self.n_fft = n_fft #Guardar las bandas MEL como atributo de clase
        #Transformar ondas a un espectograma
        self.stft_transform = torchaudio.transforms.Spectrogram( #Transformación para convertir auido en espectrogramas dconvencionales
            n_fft=n_fft, #Número de puntos para la FFT n_fft/sample_rate, resolución de la frecuencia
            win_length = None,
            hop_length=512, #Pasos
            power=2 #Potencia del espectrograma (2 E, 1 A)
        )
        #Amplitudes a escala logarítmica de decibelios
        self.db_transform = torchaudio.transforms.AmplitudeToDB() #Transformacion para convertir a decibelios(log)

    def process(self, waveform, sample_rate):
        """
        Procesa una forma de onda: re-muestrea, normaliza y transforma
        en un espectrograma STFT en escala logarítmica (dB).
        """
        #Poner todo en una frecuencia, audio crudo a misma frecuencia
        if sample_rate != self.sample_rate: #Asegurar que todos los audios tengan la misma frecuencia
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate) #Convertir las muestras, compara las frecuencias de muestra con self.sample_rate 
            waveform = resampler(waveform) #Si varía, hace un resample y aplica de nuevo el muestreo
        #Cambiar a monoaudio
        if waveform.shape[0] > 1: #De 2 canales a 1
            waveform = torch.mean(waveform, dim=0, keepdim=True) #Si ess estereo (2) promedia entre ambos, mantiene la dimensión de los canales ([1, muestras])
        #Definir una duración    
        target_samples = int(self.sample_rate * MAX_LEN) #Asegura que los audios tengan la misma duración
        if waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples] #Si es más largo, lo corta para que sea de la misma longitud
        else:
            pad_amount = target_samples - waveform.shape[1] #Si es más corto agrega esto
            waveform = F.pad(waveform, (0, pad_amount)) #Añade 0 al final de la señal
        #Normaliza la amplitud, genera el espectrograma, escala log, normalizar estadística
        waveform = waveform / waveform.abs().max() #Normalizar
        stft = self.stft_transform(waveform) #Aplicar stft
        stft_db = self.db_transform(stft) #Convierte a escala log (dB)
        stft_db = (stft_db - stft_db.mean()) / (stft_db.std() + 1e-8) #Estandarizar los valores, retorna el especrtograma normalizado 
        return stft_db  # [1, n_fft//2 + 1, time_frames]

# 2. Dataset
class InstrumentDataset(Dataset):
    """
    Dataset personalizado para audios de instrumentos musicales,
    organizados por carpetas (una por clase).

    Carga archivos .mp3 o .wav, los transforma a espectrogramas de Mel,
    y opcionalmente aplica aumento de datos.

    Atributos:
        processor (AudioProcessor): Instancia para procesar los audios.
        augment (bool): Si se debe aplicar aumento de datos o no.
        samples (list): Lista de tuplas (ruta_audio, etiqueta).
        label_map (dict): Mapeo de índice a nombre de clase.
        inverse_map (dict): Mapeo de nombre a índice de clase.
    """
    def __init__(self, data_dir, augment=False, max_samples_per_class=200): #Augment: aumento dedatos, max límite de audios por clase
        """
        Inicializa el dataset escaneando las carpetas de clases.

        Args:
            data_dir (str): Ruta al directorio que contiene subcarpetas
                            (cada una representa una clase).
            augment (bool): Si se debe aplicar aumento de datos.
            max_samples_per_class (int): Número máximo de archivos por clase.
        """
        self.processor = AudioProcessor(SAMPLE_RATE) #Crea una instancia, para convertir audios en espectrogramas
        self.augment = augment #Inicializar atributos
        self.samples = [] #Lista de tuplas: [(ruta_audio, etiqueta),]
        self.label_map = {} #Diccionario: {índice_clase: nombre_clase}
        self.inverse_map = {}  #Diccionario: {nombre_clase: índice_clase}
        
        #Analizar subcarpetas
        class_dirs = sorted(glob(os.path.join(data_dir, "*"))) #Mirar los subdirectorios de la carpeta que tiene data_dir, cada uno es una clase. Sorted para orden alfabético
        for class_idx, class_dir in enumerate(class_dirs): #Índice numérico a cada clase
            class_name = os.path.basename(class_dir) #Qué nombre tiene cada uno
            self.label_map[class_idx] = class_name #índices a nombres
            self.inverse_map[class_name] = class_idx #Nombres a índices
            
            #Para que lea mp3 o wav
            audio_files = glob(os.path.join(class_dir, "*.mp3")) + glob(os.path.join(class_dir, "*.wav")) #Buscar archivos .mp3 y .wav
            audio_files = audio_files[:max_samples_per_class] #Límitar número de muestras por clase, para tener un balance
            
            for audio_file in tqdm(audio_files, desc=f"Cargando {class_name}"): #tqdm para una barra de progreso y que se vea lindo
                self.samples.append((audio_file, class_idx)) #Almacenar tuplas con su ruta y etiqueta
        
        print(f"\nDataset cargado: {len(self.samples)} muestras, {len(self.label_map)} clases")

    def __len__(self):
        return len(self.samples) #Devolver las muestras en el dataset y pytorch sepa cuántos elementos tiene

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
            spec = self.processor.process(waveform, sample_rate) #Procesa el audio crudo en espectrograma Mel
            
            #Aumentar datos para entrenar
            if self.augment and np.random.random() < 0.5: #Aumenta los datos si hay 50% de probabilidad de usar un desplazamiento temporal en el espectrograma, para reconocer patrones independiente de la posición temporal
                spec = torch.roll(spec, shifts=np.random.randint(-10, 10), dims=2) #Desplzar el espectrograma por el eje temporal
            #Asegurar las dimensiones
            if spec.dim() == 2: #ASegurar que el espectrograma tenga tres dimensiones (canal, n_fft, tiempo)
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
        counts = np.bincount([label for _, label in self.samples])  #Cuenta número de uestras por clase, extrae las etiquetas y cuneta cuántas veces aparece
        counts = np.where(counts == 0, 1, counts) #Evitar divisiones entre 0, reemplazar 0 por 1
        weights = 1. / counts # calcula pesos inversamente proporcionales a la frecuencia de cada clase, a menor muestra mayor peso
        weights = weights / weights.sum() * len(weights) #Normalizar
        return torch.tensor(weights, dtype=torch.float32) #Convierte los pesos a un tensor para usar como funicón de perdida

# 3. Modelo CNN (ajustado para STFT)
class InstrumentCNN(nn.Module):
    """
    Modelo CNN adaptado para espectros STFT.
    La principal diferencia es que la entrada ahora tiene n_fft//2 + 1 bandas de frecuencia
    en lugar de n_mels.
    """
    def __init__(self, num_classes):
        super().__init__() #Llama nn.Module
        self.features = nn.Sequential( #Bloque de capas para procesar el espectrograma
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), #Convierte de 1 canal el espectrograma, a 32 mapas de características
            nn.BatchNorm2d(32), #Normalizar, estabiliza el entrenamiento
            nn.ReLU(), #Función de activación
            nn.MaxPool2d(2), #Reducción resolución a regiones 2x2
            nn.Dropout(0.2),  #Desactiva neuronas aleatoriamente con probabilidad de 20%, previene reajustes
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), #Ahora con 64 canales
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #Ahora con 128 canales
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            
            nn.AdaptiveAvgPool2d((1, 1)) #Convierte cualquier tamaño espacial de entrada a (1,1), por si el espectrograma tiene diferente longitud
        )
        self.classifier = nn.Sequential( 
            nn.Linear(128, 256), #Proyecta 128 características a 256 neuronas
            nn.ReLU(),
            nn.Dropout(0.5), #Regularizar
            nn.Linear(256, num_classes) #Produce un vector de puntuación por clase
        )

    def forward(self, x):
        """
        Propagación hacia adelante.

        Args:
            x (Tensor): Tensor de entrada [batch_size, 1, n_mels, tiempo].

        Returns:
            Tensor: Logits por clase.
        """
        if x.dim() == 3: #Si la entrada es de 3 dimensiones [batch, channel, n_mels, time] 
            x = x.unsqueeze(1) #pasa a [batch, 1, n_mels, time]
        x = self.features(x) #Capas convolucionales, extraer características  [batch, 128, 1, 1]
        x = x.view(x.size(0), -1) #Compacta el resultado que entra a classifier  [batch, 128]
        return self.classifier(x) #Pasa por cada capa para obtener las predicciones

# 4. Visualización
class TrainingVisualizer:
    """
    Clase para visualizar el entrenamiento del modelo, incluyendo:

    - Evolución de la pérdida (loss) y precisión (accuracy).
    - Matriz de confusión.
    - Reporte de clasificación.

    Args:
        label_map (dict): Diccionario con el mapeo de índices a nombres de clases.
    """
    def __init__(self, label_map):
        self.label_map = label_map #Guardar el mapeo de índices a nombres de clase
        self.train_loss = [] #Listas para perdidas de entrenamiento y validación
        self.val_loss = []
        self.train_acc = [] #Listas para precisión de entrenamiento y validación
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
        #Almacenar
        self.train_loss.append(tr_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(tr_acc)
        self.val_acc.append(val_acc)
        
        if epoch % 5 == 0 or epoch == EPOCHS - 1: #Actualiza valore de la época actual, cada 5 o la actual 
            self._plot_metrics() #Gráfica de evolución
            self._plot_confusion_matrix(model, val_loader) #Matriz de confusión
    #Toca cambiar los colores
    def _plot_metrics(self):
        """Genera y guarda un gráfico de la evolución de pérdida y precisión."""
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label='Entrenamiento', color="#551C8A",  linestyle=":")
        plt.plot(self.val_loss, label='Validación', color="#8A1C60DA",  linestyle="-")
        plt.title('Evolución de la Pérdida')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc, label='Entrenamiento', color="#391C8A",  linestyle=":")
        plt.plot(self.val_acc, label='Validación', color="#8A1C3DDA",  linestyle="-")
        plt.title('Evolución de la Precisión')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
    
    def _plot_confusion_matrix(self, model, loader):
        """
        Genera y guarda la matriz de confusión junto con un reporte de clasificación.

        Args:
            model (nn.Module): Modelo entrenado.
            loader (DataLoader): Loader del conjunto de validación/test.
        """
        model.eval() #Modelo de evaluación
        all_preds = [] #Guardar predicciones
        all_labels = [] #Etiquetas reales
        #Predicciones
        with torch.no_grad(): #No calcular gradientes, no se entrena, solo se evalúa 
            for inputs, labels in loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs) #Pasar los datos del loader por el modelo
                _, preds = torch.max(outputs, 1) #Guardar predicciones y etiquetas reales
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        #Matriz de confusión
        cm = confusion_matrix(all_labels, all_preds) #Crear matriz de confusión
        #cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalizar
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_map.values(), yticklabels=self.label_map.values())
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Verdaderos')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        print("\nReporte de Clasificación:")
        print(classification_report(all_labels, all_preds, target_names=self.label_map.values()))
        #Precision, exactitud al predecir Precision= true pos / (true pos + false pos)
        #Recall, capacidad para detectar los true positives de una clase Recall = true pos / (true pos + false neg)
        #f1-score, balance entre las anteriores, f1 = 2 * (precision*recall) / (precision+recall)
        #Support, número de muestras reales de cada clase por conjunto de prueba, para identificar resultados si quedan sesgados por clase 
# 5. Funciones de Entrenamiento
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, label_map):
    """
    Entrena el modelo CNN y guarda los mejores pesos según la precisión en validación.

    Args:
        model (nn.Module): Modelo a entrenar.
        train_loader (DataLoader): Datos de entrenamiento.
        val_loader (DataLoader): Datos de validación.
        criterion (Loss): Función de pérdida.
        optimizer (Optimizer): Optimizador.
        scheduler (LRScheduler): Planificador de tasa de aprendizaje.
        num_epochs (int): Número de épocas.
        label_map (dict): Mapeo de clases para visualización.

    Returns:
        nn.Module: Modelo entrenado con los mejores pesos.
    """
    visualizer = TrainingVisualizer(label_map) #Objeto para graficar
    best_acc = 0.0 #Almacenar la mejor precisión
    epoch_times = [] #Tiempos por época
    
    for epoch in range(num_epochs):
        start_time = time.time() #Iniciar temporizador
        model.train() #Poner el modelo a entrenar
        running_loss = 0.0 #Acumular las pérdida por época
        running_corrects = 0 #Acumula aciertos por época
        
        for inputs, labels in train_loader: #De CPU a GPU
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad() #Reiniciar gradientes
            outputs = model(inputs) #Calcular la salida
            loss = criterion(outputs, labels) #Calcular pérdida
            loss.backward() #Backpropagation, ajuste de los pesos del modelo minimizando el error, calcula los gradientes de pérdida respecto a cada peso
            optimizer.step() #Actualizar pesos
            
            _, preds = torch.max(outputs, 1) #ïndice de clase con mayor puntuación
            running_loss += loss.item() * inputs.size(0) #acumulación de perdida por el tamañ del batch
            running_corrects += torch.sum(preds == labels.data) #Suma de aciertos
        
        train_loss = running_loss / len(train_loader.dataset) #Promedio y precisión spbre todo el grupo de entrenamiento de pérdida
        train_acc = running_corrects.double() / len(train_loader.dataset) #Promedio y precisión spbre todo el grupo de entrenamiento de acierto
        
        #Validar
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad(): #Desactivar cálculo de gradientes
            for inputs, labels in val_loader: #Procesar cada batch
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs) #Predicciones del modelo
                loss = criterion(outputs, labels) #Calcula perdida entre outputs y labels reales
                #Acumuluar pérdidas y aciertos
                _, preds = torch.max(outputs, 1) #INdice con mayor puntucacion y predicciones finales
                val_loss += loss.item() * inputs.size(0) #Perdida del batch por numero de muestra
                val_corrects += torch.sum(preds == labels.data) #Compara predicción con real
        
        val_loss = val_loss / len(val_loader.dataset) #Calcula pérdidas
        val_acc = val_corrects.double() / len(val_loader.dataset) #Calcula aciertos
        
        if scheduler:
            scheduler.step(val_loss) #Ajusta learning rate de acuerdo a lo puesto y las pe´rdidas si es necesario
        
        if val_acc > best_acc: #si  mejoran los aciertos, los guarda en el .pth
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        epoch_time = time.time() - start_time #Calcula timepo de la época
        epoch_times.append(epoch_time)
        avg_time = sum(epoch_times) / len(epoch_times) #Tiempo promedio
        remaining_time = avg_time * (num_epochs - epoch - 1) #Tiempo restante para acabes entrenamineto
        
        visualizer.update(epoch, train_loss, val_loss, train_acc.item(), val_acc.item(), model, val_loader) #Para actualizar gráficas y métricas
        
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
    full_dataset = InstrumentDataset(data_dir, augment=True) #Instancia de dataser con aumentos, carga los archivos y etiquetas
    
    indices = list(range(len(full_dataset))) #Lista con índices
    labels = [full_dataset.samples[i][1] for i in indices] #Lista con etiquetas
    
    train_indices, val_indices = train_test_split(indices, test_size=val_split, stratify=labels) #divición índices de entrenamiento y validación, con proporciones similares
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices) #Subconjutnos indices de entrenamiento
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices) #Subconjutnos indices de validación
    
    train_labels = [full_dataset.samples[i][1] for i in train_indices] #Etiquetas entrenamiento
    class_weights = full_dataset.get_class_weights() #Calcula pesos inversamente proporcional a la frecuencia de la clase
    sample_weights = class_weights[train_labels] #Asigna a cada muestra su peso en la clase
    
    if (sample_weights <= 0).any(): #Asegurar ningún peso 0
        sample_weights = torch.clamp(sample_weights, min=1e-8) #Forzar valor mínimo
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True) #Selección muestras de forma porporcional a los pesos, se pueden repetir muestras
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True) #Dataloadre para entrenamiento, sampler para balancear, num_workers hilos para cargar en paralelo, pin para acelelrar transferencia de GPU a CPU
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) #Dataloader validación, shuffle para no mezclar orden de validación 
    
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
        predict_audio.processor = AudioProcessor(SAMPLE_RATE) #Genera espectrrograma
    
    try:
        waveform, sample_rate = torchaudio.load(file_path) #Cargar archivo de audio
        spec = predict_audio.processor.process(waveform, sample_rate) #Convertir onda en espectrograma de decibelios
         #estos if son para agrgar las dimensiones en caso de ser necesario
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
            outputs = predict_audio.model(spec) #Obtiene logits aka vectores
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
